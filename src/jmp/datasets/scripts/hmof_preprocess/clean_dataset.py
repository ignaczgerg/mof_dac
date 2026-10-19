import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('/ibex/project/c2261/datasets/hmof/raw/all_MOFs_screening_data.csv')

    properties = [
                "CO2_uptake_P0.15bar_T298K [mmol/g]",
                "heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]",
                "excess_CO2_uptake_P0.15bar_T298K [mmol/g]",
                "CO2_binary_uptake_P0.15bar_T298K [mmol/g]",
                "heat_adsorption_CO2_binary_P0.15bar_T298K [kcal/mol]",
                "excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]",
                "N2_binary_uptake_P0.85bar_T298K [mmol/g]",
                "heat_adsorption_N2_binary_P0.85bar_T298K [kcal/mol]",
                "excess_N2_binary_uptake_P0.85bar_T298K [mmol/g]",
                "CO2/N2_selectivity",
            ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=properties, inplace=True)
    df = df[(df[properties] != 0).all(axis=1)]
    logs = df.copy()
    for p in properties:
        try:
            logs["ln_"+str(p)] = np.log(df[p])
        except Exception as e:
            print(f"Error processing {p}: {e}")

    logs = logs[logs['ln_CO2_uptake_P0.15bar_T298K [mmol/g]'] > -4]
    logs = logs[logs["ln_CO2_binary_uptake_P0.15bar_T298K [mmol/g]"] > -4]
    logs = logs[logs["ln_N2_binary_uptake_P0.85bar_T298K [mmol/g]"] > -4]
    
    df = logs.copy()
    cleaned = df[properties].copy()
    cleaned['id'] = df['MOFname']
    print(f"Number of MOFs before cleaning: {len(df)}")
    print(f"Number of MOFs after cleaning: {len(cleaned)}")
    print(f"Kept these properties: {list(cleaned.columns)}")
    cleaned.to_csv('/ibex/project/c2261/datasets/hmof/raw/all_MOFs_screening_data_cleaned.csv', index=False)


if __name__ == "__main__":
    main()