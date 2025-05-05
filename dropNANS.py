import pandas as pd

def main():

    # 1) Load
    csv = "LTM_data/LTM_real_data/Moneyball/train/Moneyball--train--128-seed2.csv"
    df = pd.read_csv(csv)

    # 2) Drop (ignoring names that don't exist)
    df = df.drop(columns=['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG'], errors="ignore")

    # 3) Write
    df.to_csv(csv, index=False)

if __name__ == "__main__":
    main()
