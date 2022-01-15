import json
import os
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, redirect

from preprocess_truth import preprocess

app = Flask(__name__)

MAX_SIZE_MB = 1
DATA_FOLDER = "files"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

app.config["UPLOAD_FOLDER"] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_SIZE_MB * 1024 * 1024


# Fetching of new data is assumed to be performed on first eval. request after 8AM UTC
# Heroku wants my credit card info to set up a scheduled task so this contains an improvised refreshing mechanism
preprocess()
_today = datetime.now(timezone.utc)
LAST_DATA_UPDATE = datetime(_today.year, _today.month, _today.day, 8, 0, tzinfo=timezone.utc)


@app.errorhandler(413)
def request_entity_too_large(error):
    return f"The uploaded file with predictions is too large (> {MAX_SIZE_MB} MB) " \
           f"<a href='/'>Back to main page</a>", 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


def mean_regressor(true_weekly_cases, window_size):
    preds = []
    for idx_week in range(len(true_weekly_cases)):
        preds.append(
            sum(true_weekly_cases[max(0, idx_week - window_size): idx_week]) / window_size
        )

    return preds


def compute_metrics(pred_cases, first_n_weeks=None, last_n_weeks=None):
    try:
        df = pd.read_csv("covid.slovenia.week.csv")

        if first_n_weeks is not None:
            df = df.iloc[: first_n_weeks]
        if last_n_weeks is not None:
            df = df.iloc[-last_n_weeks:]
    except FileNotFoundError:
        raise FileNotFoundError("Could not find pre-processed ground-truth data at 'covid.slovenia.week.csv'")

    assert len(pred_cases) == df.shape[0], \
        f"Predictions required for {df.shape[0]} weeks, got {len(pred_cases)} predictions"

    rmse_pred = mean_squared_error(pred_cases, df["NewCases"], squared=False)
    # Mean of previous one week aka copy baseline
    mean_preds_1week = mean_regressor(df["NewCases"].values, window_size=1)
    rmse_1week = mean_squared_error(mean_preds_1week, df["NewCases"], squared=False)
    # Mean of previous four weeks
    mean_preds_4weeks = mean_regressor(df["NewCases"].values, window_size=4)
    rmse_4week = mean_squared_error(mean_preds_4weeks, df["NewCases"], squared=False)

    returned_stats = {
        "true_cases": df["NewCases"].tolist(),
        "predicted_cases": pred_cases,
        "predicted_mean@1": mean_preds_1week,
        "predicted_mean@4": mean_preds_4weeks,
        "rmse": round(rmse_pred, 3),
        "rmse_1week": round(rmse_1week, 3),
        "rmse_4week": round(rmse_4week, 3),
        "rrmse_1week": round(rmse_pred / rmse_1week, 3),
        "rrmse_4weeks": round(rmse_pred / rmse_4week, 3)
    }

    return returned_stats


@app.route('/evaluate', methods=["GET", "POST"])
def evaluate():
    if request.method == "GET":
        return redirect("/")

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return redirect("/")

    if uploaded_file.content_type != "text/plain":
        return f"The uploaded file is not a text file ('{uploaded_file.content_type}')", 400

    try:
        lines = list(map(lambda s: int(s.strip()), uploaded_file.readlines()))
    except ValueError:
        return "File contains non-numeric elements", 400

    # Poor man's cron
    today = datetime.now(timezone.utc)
    global LAST_DATA_UPDATE
    if LAST_DATA_UPDATE is None or (today - LAST_DATA_UPDATE).total_seconds() > 24 * 3600:
        print(f"Fetching fresh data @ {str(today)}")
        LAST_DATA_UPDATE = datetime(today.year, today.month, today.day, 8, 0, tzinfo=timezone.utc)
        preprocess()

    try:
        stats = compute_metrics(lines)

        retvals = {
            "full": {
                "rmse": stats["rmse"], "rrmse_1week": stats["rrmse_1week"], "rrmse_4weeks": stats["rrmse_4weeks"],
                "intermediate": json.dumps({
                    "mean@1_rmse": stats["rmse_1week"],
                    "mean@4_rmse": stats["rmse_4week"],
                    "true_cases": stats["true_cases"],
                    "predicted_cases": stats["predicted_cases"],
                    "mean@1_cases": stats["predicted_mean@1"],
                    "mean@4_cases": stats["predicted_mean@4"]
                }, indent=4)
            }
        }

        # hacky: take first 104 weeks == including epi. week ending on 15th Jan 2022 (2nd eval checkpoint)
        if today > datetime(2022, 1, 16, 8, 0, tzinfo=timezone.utc):
            jan16_full = compute_metrics(lines[:104], first_n_weeks=104)
            retvals["jan16_stats"] = {
                "full": {
                    "rmse": jan16_full["rmse"], "rrmse_1week": jan16_full["rrmse_1week"], "rrmse_4weeks": jan16_full["rrmse_4weeks"],
                    "intermediate": json.dumps({
                        "mean@1_rmse": jan16_full["rmse_1week"],
                        "mean@4_rmse": jan16_full["rmse_4week"],
                        "true_cases": jan16_full["true_cases"],
                        "predicted_cases": jan16_full["predicted_cases"],
                        "mean@1_cases": jan16_full["predicted_mean@1"],
                        "mean@4_cases": jan16_full["predicted_mean@4"]
                    }, indent=4)
                }
            }

    except AssertionError as exc:
        return str(exc), 400
    except FileNotFoundError as exc:
        return f"Server-side error: {str(exc)}", 500

    return render_template("evaluation.html", **retvals)


if __name__ == '__main__':
    app.run(port=33507)
