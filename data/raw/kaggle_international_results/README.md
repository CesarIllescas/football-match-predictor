# Kaggle international football results

Source: [International football results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
(the dataset is continuously updated despite the name).

Place these files here:

```
results.csv        (required)
shootouts.csv      (optional)
goalscorers.csv    (optional)
former_names.csv   (optional)
```

## Automatic download

```bash
python scripts/download_data.py --source kaggle_international_results
```

This requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in your `.env` (or a `kaggle.json`).

## Manual download

1. Log in to Kaggle.
2. Download the dataset zip from the link above.
3. Extract the CSV files directly into this folder.
