# BIRD Data

This release copy already includes the lightweight BIRD metadata and query
files needed by the code:

- `dev_data/db2schema.json`
- `dev_data/dev.json`
- `dev_data/dev.sql`
- `dev_data/dev_prompt.json`
- `dev_data/dev_prompt.jsonl`
- `dev_data/dev_tables.json`
- `dev_data/dev_tied_append.json`
- `train_data/db2schema.json`
- `train_data/train.json`
- `train_data/train_gold.sql`
- `train_data/train_prompt.json`
- `train_data/train_schemas.jsonl`
- `train_data/train_tables.json`

The SQLite databases are intentionally not included in the repo.

## Expected Layout

- `dev_data/dev_databases/<db_id>/<db_id>.sqlite`
- `train_data/train_databases/<db_id>/<db_id>.sqlite`
- `ground_truth_cache.pkl` is optional and only speeds up repeated SQL evals

## Official Downloads

- Official BIRD benchmark homepage: <https://bird-bench.github.io/>
- Official Mini-Dev repository: <https://github.com/bird-bench/mini_dev>
- Mini-Dev direct package: <https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip>
- Mini-Dev complete package mirror: <https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view?usp=sharing>
- Mini-Dev Hugging Face dataset: <https://huggingface.co/datasets/birdsql/bird_mini_dev>
- Official BIRD train split package: <https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip>

## How To Use With This Repo

1. Keep the metadata files already committed in this directory.
2. Use the helper script from the repo root:

```bash
# Mini-dev SQLite databases for task.split=dev
bash scripts/setup_bird_data.sh dev

# Full train SQLite databases for task.split=train
bash scripts/setup_bird_data.sh train
```

3. The helper script downloads the official archive into `data/bird/.downloads/`
   and extracts only the `dev_databases/` or `train_databases/` subtree into:
   - `data/bird/dev_data/dev_databases/`
   - `data/bird/train_data/train_databases/`

If you prefer to keep the databases elsewhere, set `LSE_BIRD_DATA_ROOT` to that
external location instead.
