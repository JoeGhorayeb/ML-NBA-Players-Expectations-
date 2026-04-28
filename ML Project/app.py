from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

team_ranking = {
    'BOS': 1, 'LAL': 2, 'SAS': 3, 'PHI': 4, 'GSW': 5,
    'CHI': 6, 'MIA': 7, 'OKC': 8, 'DET': 9, 'NYK': 10,
    'MIL': 11, 'HOU': 12, 'POR': 13, 'ATL': 14, 'CLE': 15,
    'PHX': 16, 'DAL': 17, 'DEN': 18, 'UTA': 19, 'WAS': 20,
    'IND': 21, 'TOR': 22, 'SAC': 23, 'ORL': 24, 'BKN': 25,
    'LAC': 26, 'NOP': 27, 'MEM': 28, 'MIN': 29, 'CHA': 30
}

numeric_features = [
    'sim', 'draft_expectation', 'pressure_factor', 'total_pressure',
    'pressure_scaled', 'ppg_adj', 'rpg_adj', 'apg_adj', 'spg_adj',
    'bpg_adj', 'BPM_adj', 'S', 'sb', 'pressure_scaled_norm', 'sim_norm'
]

def classify_expectation(val):
    if val >= 110:
        return 'Exceeded'
    elif val >= 75:
        return 'Met'
    else:
        return 'Underperformed'

def prepare_single_input(height_cm, weight_kg, team, draft_pick,
                         ppg, rpg, apg, spg, bpg, BPM):
    team_rank     = team_ranking.get(team)
    team_strength = (team_rank - 30) / (-29) * 0.5
    height_m      = height_cm / 100
    bmi           = weight_kg / (height_m ** 2)
    sim           = np.exp(-((bmi - 25.5) ** 2) / 50)
    draft_expectation  = (1 / draft_pick) ** 2
    pressure_factor    = 0.4 + team_strength
    total_pressure     = draft_expectation * team_strength
    pressure_scaled    = (total_pressure - data['total_pressure'].min()) / (data['total_pressure'].max() - data['total_pressure'].min())
    ppg_adj = ppg * pressure_factor
    rpg_adj = rpg * pressure_factor
    apg_adj = apg * pressure_factor
    spg_adj = spg * pressure_factor
    bpg_adj = bpg * pressure_factor
    BPM_adj = BPM * pressure_factor
    stats_cols   = ['ppg_adj', 'rpg_adj', 'apg_adj', 'spg_adj', 'bpg_adj']
    train_means  = data[stats_cols].mean()
    train_stds   = data[stats_cols].std()
    player_vals  = np.array([ppg_adj, rpg_adj, apg_adj, spg_adj, bpg_adj])
    Z_sum        = ((player_vals - train_means.values) / train_stds.values).sum()
    S            = 1 / (1 + np.exp(-Z_sum))
    sb           = 1 / (1 + np.exp(-(BPM - data['BPM'].mean()) / data['BPM'].std()))
    pressure_scaled_norm = (pressure_scaled - data['pressure_scaled'].min()) / (data['pressure_scaled'].max() - data['pressure_scaled'].min())
    sim_norm     = (sim - data['sim'].min()) / (data['sim'].max() - data['sim'].min())
    return pd.DataFrame([{
        'sim': sim, 'draft_expectation': draft_expectation,
        'pressure_factor': pressure_factor, 'total_pressure': total_pressure,
        'pressure_scaled': pressure_scaled, 'ppg_adj': ppg_adj,
        'rpg_adj': rpg_adj, 'apg_adj': apg_adj, 'spg_adj': spg_adj,
        'bpg_adj': bpg_adj, 'BPM_adj': BPM_adj, 'S': S, 'sb': sb,
        'pressure_scaled_norm': pressure_scaled_norm, 'sim_norm': sim_norm,
    }])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        sample = prepare_single_input(
            height_cm  = float(request.form['height_cm']),
            weight_kg  = float(request.form['weight_kg']),
            team       = request.form['team'],
            draft_pick = int(request.form['draft_pick']),
            ppg        = float(request.form['ppg']),
            rpg        = float(request.form['rpg']),
            apg        = float(request.form['apg']),
            spg        = float(request.form['spg']),
            bpg        = float(request.form['bpg']),
            BPM        = float(request.form['BPM'])
        )
        score  = float(ridge_model.predict(sample[numeric_features])[0])
        label  = classify_expectation(score)
        result = {'score': round(score, 2), 'label': label}
    return render_template('index.html', result=result, teams=list(team_ranking.keys()))

if __name__ == '__main__':
    app.run(debug=True)

#http://127.0.0.1:5000 run this