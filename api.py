from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Promoted teams context
PROMOTED_TEAMS_NOTE = """
Note:
- Sunderland, Leeds United, and Burnley have been promoted to the Premier League for the 2025/26 season.
- Treat all three as active Premier League teams.
- Do NOT say they are in lower divisions or refer to them as hypothetical.
"""

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load match data
df = pd.read_csv("epl_matches_2017_2023.csv", low_memory=False)

# =======================
# Form & Head-to-Head Summary Helpers
# =======================

def summarize_form(matches, team_name):
    wins = (matches['Result'] == 'W').sum()
    draws = (matches['Result'] == 'D').sum()
    losses = (matches['Result'] == 'L').sum()
    avg_goals_for = matches['GF'].mean()
    avg_goals_against = matches['GA'].mean()

    return f"""{team_name} - Last {len(matches)} Matches:
- Wins: {wins}, Draws: {draws}, Losses: {losses}
- Avg Goals Scored: {avg_goals_for:.2f}, Avg Goals Conceded: {avg_goals_against:.2f}"""

def summarize_h2h(matches, home, away):
    if matches.empty:
        return "No recent head-to-head match data available."

    total = len(matches)
    home_wins = ((matches['Team'] == home) & (matches['Result'] == 'W')).sum()
    away_wins = ((matches['Team'] == away) & (matches['Result'] == 'W')).sum()
    draws = (matches['Result'] == 'D').sum()

    return f"Head-to-Head ({home} vs {away}) - Last {total} Matches:\n- {home} Wins: {home_wins}, {away} Wins: {away_wins}, Draws: {draws}"

# =======================
# Rule-Based Prediction Logic
# =======================

def get_team_form(df, team, n=5):
    recent = df[df['Team'] == team].sort_values(by='date', ascending=False).head(n)
    wins = (recent['Result'] == 'W').sum()
    draws = (recent['Result'] == 'D').sum()
    losses = (recent['Result'] == 'L').sum()
    avg_gf = recent['GF'].mean()
    avg_ga = recent['GA'].mean()
    avg_xg = recent['xG'].mean() if 'xG' in recent else 0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_gf": avg_gf,
        "avg_ga": avg_ga,
        "avg_xg": avg_xg,
    }

def get_head_to_head(df, home, away, n=5):
    h2h = df[((df['Team'] == home) & (df['Opponent'] == away)) |
             ((df['Team'] == away) & (df['Opponent'] == home))]\
             .sort_values(by='date', ascending=False).head(n)
    home_wins = ((h2h['Team'] == home) & (h2h['Result'] == 'W')).sum()
    away_wins = ((h2h['Team'] == away) & (h2h['Result'] == 'W')).sum()
    draws = (h2h['Result'] == 'D').sum()
    return {"home_wins": home_wins, "away_wins": away_wins, "draws": draws}

def rule_based_prediction(home_team, away_team):
    home = get_team_form(df, home_team)
    away = get_team_form(df, away_team)
    h2h = get_head_to_head(df, home_team, away_team)

    # Simple scoring system
    home_score = (
        home['wins']
        + (1 if home['avg_gf'] > away['avg_ga'] else 0)
        + (1 if home['avg_xg'] > away['avg_xg'] else 0)
        + h2h['home_wins']
    )
    away_score = (
        away['wins']
        + (1 if away['avg_gf'] > home['avg_ga'] else 0)
        + (1 if away['avg_xg'] > home['avg_xg'] else 0)
        + h2h['away_wins']
    )

    diff = home_score - away_score

    if diff >= 2:
        return f"{home_team} to win"
    elif diff <= -2:
        return f"{away_team} to win"
    else:
        return "Draw"

# =======================
# Prediction Endpoint
# =======================

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if not home_team or not away_team:
        return {"error": "Missing team names."}

    prediction = rule_based_prediction(home_team, away_team)

    home_form = summarize_form(df[df['Team'] == home_team].sort_values(by='date', ascending=False).head(10), home_team)
    away_form = summarize_form(df[df['Team'] == away_team].sort_values(by='date', ascending=False).head(10), away_team)
    h2h_matches = df[((df['Team'] == home_team) & (df['Opponent'] == away_team)) |
                     ((df['Team'] == away_team) & (df['Opponent'] == home_team))].sort_values(by='date', ascending=False).head(10)
    h2h_form = summarize_h2h(h2h_matches, home_team, away_team)

    discussion = (
        f"{home_team} comes into this fixture with {home_form.splitlines()[1]}. "
        f"{away_team} shows {away_form.splitlines()[1]}. "
        f"The head-to-head record suggests: {h2h_form.splitlines()[-1]}. "
        f"Considering form and past performance, this match is predicted to end in: {prediction.lower()}."
    )

    return {
        "prediction": prediction,
        "stats": [home_form, away_form, h2h_form],
        "discussion": discussion.strip()
    }
