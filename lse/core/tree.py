from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import math
import copy
import json
import networkx as nx
import matplotlib.pyplot as plt



@dataclass
class TreeNode:
    id: int
    parent: Optional["TreeNode"]
    children: List["TreeNode"] = field(default_factory=list)
    depth: int = 0
    visits: int = 0
    performance: float = float("nan")
    round_idx: Optional[int] = None
    instructions: str = ""
    conversation: Optional[List[Dict[str, str]]] = None
    summary: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[List[Dict[str, str]]]] = None
    self_evolve_prompt: Optional[List[Dict[str, str]]] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"TreeNode(id={self.id}, round_idx={self.round_idx}, parent={self.parent.id if self.parent else None}, depth={self.depth}, visits={self.visits}, performance={self.performance})"


class EvolutionTree:
    """
    Simple in-memory tree for storing round histories and metadata, and
    selecting a node to evolve from.

    Selection criteria:
      - deepest: choose the deepest leaf (ties broken by highest id)
      - best: choose the leaf with highest performance (ties broken by depth then id)
      - ucb: Upper Confidence Bound over leaves using stored performance as mean
    """

    def __init__(self, selection: str = "deepest", ucb_c: float = 2.0, base_instructions: str = ""):
        self.root = TreeNode(
            id=0,
            parent=None,
            depth=0,
            visits=0,
            performance=float("nan"),
            round_idx=0,
            instructions=base_instructions,
            summary=None,
            history=None,
        )
        self._next_id = 1
        self.selection = selection
        self.ucb_c = float(ucb_c)
        self._total_selections = 0
        self.nodes: Dict[int, TreeNode] = {0: self.root}

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def update_node(self, id_or_node: int | TreeNode, **kwargs) -> None:
        if isinstance(id_or_node, TreeNode):
            node = id_or_node
        else:
            node = self.nodes[id_or_node]
        for k, v in kwargs.items():
            if not hasattr(node, k):
                raise ValueError(f"Node {id_or_node} has no attribute {k}")
            setattr(node, k, v)


    def add_node(
        self,
        parent: TreeNode,
        instructions: str,
        round_idx: int,
        conversation: List[Dict[str, str]]
    ) -> TreeNode:
        node = TreeNode(
            id=self._new_id(),
            parent=parent,
            depth=(0 if parent is None else parent.depth + 1),
            visits=0,
            performance=float("nan"),
            round_idx=round_idx,
            instructions=instructions,
            conversation=conversation,
            summary=None,
            history=None,
        )
        if parent is not None:
            parent.children.append(node)
        self.nodes[node.id] = node
        return node


    @property
    def node_list(self) -> List[TreeNode]:
        return list(self.nodes.values())

    def calculate_improvement(self):
        for node in self.node_list:
            if node.parent is not None and not math.isnan(node.performance):
                node.improve_from_parent = node.performance - node.parent.performance
            else:   
                node.improve_from_parent = -math.inf

    def _select_deepest(self) -> Optional[TreeNode]:
        nodes = self.node_list
        if not nodes:
            return None
        # deepest, then greatest id (most recent)
        nodes.sort(key=lambda n: (n.depth, n.id), reverse=True)
        return nodes[0]

    def _select_best(self) -> Optional[TreeNode]:
        nodes = self.node_list
        if not nodes:
            return None
        def perf(n: TreeNode) -> float:
            return -math.inf if math.isnan(n.performance) else n.performance
        nodes.sort(key=lambda n: (perf(n), n.depth, n.id), reverse=True)
        return nodes[0]

    def _select_ucb(self) -> Optional[TreeNode]:
        nodes = self.node_list
        if not nodes:
            return None
            
        total = max(1, self._total_selections)
        def score(n: TreeNode) -> float:
            mean = -math.inf if math.isnan(n.performance) else n.performance
            bonus = self.ucb_c * math.sqrt(math.log(total) / max(1, n.visits))
            return mean + bonus
        nodes.sort(key=lambda n: score(n), reverse=True)
        return nodes[0]

    def select(self, criterion: Optional[str] = None) -> Optional[TreeNode]:
        crit = (criterion or self.selection or "deepest").lower()
        if crit == "deepest":
            chosen = self._select_deepest()
        elif crit == "best":
            chosen = self._select_best()
        elif crit == "ucb":
            chosen = self._select_ucb()
        else:
            raise ValueError(f"Invalid selection criterion: {criterion}")
        if chosen is not None:
            chosen.visits += 1
            self._total_selections += 1
        return chosen

    def size(self) -> int:
        # number of nodes in the tree (including root)
        count = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children)
        return count

    def calculate_improve_potential(self):
        # improve potential = max(performance) - performance
        max_performance = max(node.performance for node in self.node_list)
        for node in self.node_list:
            node.improve_potential = max_performance - node.performance

    def __len__(self) -> int:
        return self.size()

    # ---------- Rendering / Export ----------

    def save_tree(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "tree.json")

        def node_payload(n: TreeNode) -> Dict[str, Any]:
            return {
                "id": n.id,
                "parent_id": (n.parent.id if n.parent is not None else None),
                "depth": n.depth,
                "visits": n.visits,
                "performance": (None if math.isnan(n.performance) else float(n.performance)),
                "round_idx": n.round_idx,
                "instructions": n.instructions,
                "conversation": n.conversation,
                "summary": n.summary,
                "history": n.history,
                "self_evolve_prompt": n.self_evolve_prompt,
            }

        data: Dict[str, Any] = {
            "format_version": 1,
            "selection": self.selection,
            "ucb_c": self.ucb_c,
            "next_id": self._next_id,
            "total_selections": self._total_selections,
            "root_id": self.root.id,
            "nodes": [node_payload(n) for n in sorted(self.node_list, key=lambda x: x.id)],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_tree(cls, path_or_dir: str) -> "EvolutionTree":
        """Load an EvolutionTree from a JSON file produced by save_tree.

        Args:
            path_or_dir: Path to a tree.json file or a directory containing it.

        Returns:
            Reconstructed EvolutionTree instance.
        """
        if os.path.isdir(path_or_dir):
            json_path = os.path.join(path_or_dir, "tree.json")
        else:
            json_path = path_or_dir

        with open(json_path, "r") as f:
            data = json.load(f)

        selection = data.get("selection", "deepest")
        ucb_c = float(data.get("ucb_c", 2.0))
        root_id = int(data.get("root_id", 0))

        nodes_payload = data.get("nodes", [])
        payload_by_id: Dict[int, Dict[str, Any]] = {int(p["id"]): p for p in nodes_payload}
        root_payload = payload_by_id.get(root_id, None)
        base_instructions = (root_payload or {}).get("instructions", "")

        tree = cls(selection=selection, ucb_c=ucb_c, base_instructions=base_instructions)

        # Build nodes without links first
        id_to_node: Dict[int, TreeNode] = {}
        for nid, p in sorted(payload_by_id.items(), key=lambda kv: kv[0]):
            perf_val = float("nan") if p.get("performance") is None else float(p.get("performance"))
            node = TreeNode(
                id=int(p["id"]),
                parent=None,
                children=[],
                depth=int(p.get("depth", 0)),
                visits=int(p.get("visits", 0)),
                performance=perf_val,
                round_idx=(p.get("round_idx") if p.get("round_idx") is None else int(p.get("round_idx"))),
                instructions=p.get("instructions", ""),
                conversation=p.get("conversation"),
                summary=p.get("summary"),
                history=p.get("history"),
                self_evolve_prompt=p.get("self_evolve_prompt"),
            )
            id_to_node[node.id] = node

        # Second pass: wire up parents and children
        for nid, p in payload_by_id.items():
            node = id_to_node[nid]
            parent_id = p.get("parent_id")
            if parent_id is not None:
                parent_node = id_to_node[int(parent_id)]
                node.parent = parent_node
                parent_node.children.append(node)

        # Finalize tree fields
        tree.root = id_to_node[root_id]
        tree.nodes = id_to_node
        tree._next_id = int(data.get("next_id", (max(id_to_node.keys()) + 1 if id_to_node else 1)))
        tree._total_selections = int(data.get("total_selections", 0))

        return tree

    def _node_label(self, n: TreeNode) -> str:
        perf_str = "nan" if math.isnan(n.performance) else f"{n.performance:.3f}"
        ridx = "-" if n.round_idx is None else str(n.round_idx)
        return f"id={n.id} r#{ridx} acc={perf_str} depth={n.depth} visits={n.visits}"

    def render_text(self) -> str:
        lines: List[str] = []

        def dfs(node: TreeNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            if node is self.root:
                lines.append(self._node_label(node))
            else:
                lines.append(prefix + connector + self._node_label(node))
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, c in enumerate(node.children):
                dfs(c, child_prefix, i == len(node.children) - 1)

        dfs(self.root)
        return "\n".join(lines)



    def save_visualizations(self, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        ascii_path = os.path.join(out_dir, "tree.txt")
        with open(ascii_path, "w") as f:
            f.write(self.render_text())
        paths = {"ascii": ascii_path}
        # Optional: save a NetworkX plot if dependencies are available
        try:
            nx_png = os.path.join(out_dir, "tree_networkx.png")
            self._save_networkx_plot(nx_png)
            paths["networkx_png"] = nx_png
        except Exception as _e:
            print(f"Warning: Failed to save networkx plot: {_e}")
            pass
        return paths

    def _node_plot(self, n: TreeNode) -> str:
        perf_str = "nan" if math.isnan(n.performance) else f"{n.performance:.2f}"
        ridx = "-" if n.round_idx is None else str(n.round_idx)
        return f"r#{ridx} {perf_str}"


    def _save_networkx_plot(self, save_path: str) -> None:
        G = nx.DiGraph()

        # Build nodes and edges
        stack = [self.root]
        while stack:
            n = stack.pop()
            G.add_node(n.id, label=self._node_plot(n), depth=n.depth)
            for c in n.children:
                G.add_edge(n.id, c.id)
            stack.extend(n.children)

        # Assign x positions to leaves left-to-right, then parents at midpoints
        positions: Dict[int, Any] = {}

        def assign_positions(node: TreeNode, next_x: float) -> float:
            if not node.children:
                positions[node.id] = (next_x, -node.depth)
                return next_x + 1
            x = next_x
            child_xs = []
            for c in node.children:
                x = assign_positions(c, x)
                child_xs.append(positions[c.id][0])
            positions[node.id] = ((child_xs[0] + child_xs[-1]) / 2.0, -node.depth)
            return x

        assign_positions(self.root, 0)

        # Draw
        plt.figure(figsize=(max(6.0, len([n for n in G.nodes if G.out_degree(n) == 0]) * 0.8),  
                            max(4.0, (max((d for _, d in positions.values()), default=0) * -1 + 2) * 0.9)))
        nx.draw_networkx_edges(G, pos=positions, arrows=False, edge_color="#444444", width=1.2)
        nx.draw_networkx_nodes(G, pos=positions, node_size=260, node_color="#2e6fbb")
        labels = {nid: G.nodes[nid]["label"] for nid in G.nodes}
        nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()



