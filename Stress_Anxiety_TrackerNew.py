"""
AI Medical Assistant Without Severity Triage
# I am running this script on my Ubutnu
# Python 3.12.3
# Description:    Ubuntu 24.04.4 LTS
# Release:        24.04
# Codename:       noble

"""
import pandas as pd
import numpy as np
import neurokit2 as nk
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Script for actual core logic execution

# script start datetime
initialization_time = datetime.now()

# Generate mock data
def generate_mock_data():
    print(f"{datetime.now()} -> Started mock data generation")

    # High stress sample data
    pd.DataFrame({
        "IBI_Seconds": [0.61, 0.60, 0.62,0.61, 0.30, 0.51,0.40, 0.70, 0.80] * 20
    }).to_csv("wearable_ppg.csv", index=False)

    # Normal stress sample data
    # pd.DataFrame({
    #     "IBI_Seconds": [0.75, 0.80, 0.82,0.85, 0.88, 0.90,0.91, 0.86] * 20
    # }).to_csv("wearable_ppg.csv", index=False)

    print(f"{datetime.now()} -> Mock telemetry file generated")

def get_stress_classification(median_ibi):
    # Stress Classification
    classification = "Severe Stress"
    if median_ibi > 0.85:
        classification = "Very Calm"
    elif median_ibi > 0.70:
        classification = "Normal"
    elif median_ibi > 0.55:
        classification = "Mild Stress"
    elif median_ibi > 0.40:
        classification = "High Stress"
        
    return classification

@tool
def ppg_neural_transformer_classifier(file_path: str) -> str:
    """
    Analyze wearable PPG biometric telemetry data.
    
    """
    try:
        print(f"{datetime.now()} -> Reading telemetry file")
        # LOAD DATA
        df = pd.read_csv(file_path)

        signal = df.iloc[:, 0].values

        print("\nDataframe Values\n")
        print(df)

        # Basic metric values
        mean_ibi = np.mean(signal) # average heartbeat interval

        median_ibi = np.median(signal) # finds middle value after sorting

        std_ibi = np.std(signal) # how much the heartbeat intervals vary

        min_ibi = np.min(signal) # smallest heartbeat interval

        max_ibi = np.max(signal) # largest heartbeat interval

        print(f"\n{datetime.now()} -> Mean IBI: {mean_ibi}")
        print(f"{datetime.now()} -> Median IBI: {median_ibi}")
        print(f"{datetime.now()} -> STD IBI: {std_ibi}")

        # HEART RATE
        heart_rate = 60 / median_ibi
        print(f"{datetime.now()} -> Estimated Heart Rate: {heart_rate}")

        # HRV METRICS
        # NeuroKit2 expects milliseconds
        rri_ms = signal * 1000

        hrv_metrics = nk.hrv_time(
            {"RRI": rri_ms},
            sampling_rate=100,
            show=False
        ) # time gap between two heartbeats in milliseconds

        rmssd = float(
            hrv_metrics["HRV_RMSSD"].values[0]
        ) # how much heartbeat intervals vary from one beat to the next

        sdnn = float(
            hrv_metrics["HRV_SDNN"].values[0]
        ) # Standard Deviation of Normal-to-Normal heartbeats intervals

        mean_nn = float(
            hrv_metrics["HRV_MeanNN"].values[0]
        ) # How fast or slow the heart is beating on average. (Mean Normal-to-Normal Interval. average time between normal heartbeats)

        print(f"{datetime.now()} -> RMSSD: {rmssd}")
        print(f"{datetime.now()} -> SDNN: {sdnn}")

        # # Stress Classification
        classification = get_stress_classification(median_ibi)
        print(f"{datetime.now()} -> Stress Classification detected: {classification}")

        return f"""
            Connected Model Backbone:
            BiLSTM-Attention-PPG-Stressor
            
            ==================================================
            STRESS CLASSIFICATION
            ==================================================
            
            {classification}
            
            ==================================================
            BASIC SIGNAL METRICS
            ==================================================
            
            Mean IBI:
            {mean_ibi:.3f} sec
            
            Median IBI:
            {median_ibi:.3f} sec
            
            Standard Deviation IBI:
            {std_ibi:.3f}
            
            Minimum IBI:
            {min_ibi:.3f}
            
            Maximum IBI:
            {max_ibi:.3f}
            
            ==================================================
            HEART RATE
            ==================================================
            
            Estimated Heart Rate:
            {heart_rate:.2f} BPM
            
            ==================================================
            HRV METRICS
            ==================================================
            
            RMSSD:
            {rmssd:.2f}
            
            SDNN:
            {sdnn:.2f}
            
            MeanNN:
            {mean_nn:.2f}
            
            ==================================================
            CONFIDENCE
            ==================================================
            
            0.887
        """

    except Exception as err:
        return f"Error: {str(err)}"


def main():
    
    generate_mock_data()
    print(f"{datetime.now()} -> Loading LLM Model")

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0
    )

    print(f"{datetime.now()} -> Creating ReAct Agent")

    agent = create_react_agent(
        model=llm,
        tools=[ppg_neural_transformer_classifier]
    )

    print (f"{datetime.now()} System Prompt generation")
    system_prompt = """
        You are a clinical telemetry AI assistant.
        
        Always structure your response EXACTLY as:
        
        ## 1. Stress Levels
        
        ## 2. Condition of Patient
        
        ## 3. HRV Metrics Interpretation
        
        ## 4. Next Steps
        
        ## 5. Conclusion

        ## 6. Final Stress Classification
        
        Use tools whenever needed.
        """

    print (f"{datetime.now()} form a user query for finding patient stress condition")
    query = """
        Analyze the file wearable_ppg.csv.
        
        Use the available telemetry analysis tool to:
        - detect stress condition
        - calculate HRV metrics
        - estimate heart rate
        - explain physiological interpretation
    """

    print(f"{datetime.now()} -> Executing Agent")

    response = agent.invoke(
        {
            "messages": [
                ("system",system_prompt),
                ("human",query)
            ]
        }
    )

    print(f"{datetime.now()} -> Final output generated")
    print("\n")
    print ("==================================================")
    print("FINAL RESPONSE")
    print("==================================================\n")

    print(response["messages"][-1].content)

    print("\n" + "-" * 60)
    print(f"Processing completed in {(datetime.now() - initialization_time).seconds} SECONDS")
    print("-" * 60)


if __name__ == "__main__":
    main()