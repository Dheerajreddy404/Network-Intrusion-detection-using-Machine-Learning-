import pandas as pd
import os

def process_file(file, label):
    print(f"Processing {file}...")

    # Load CSV
    df = pd.read_csv(file)

    # Rename Wireshark export columns
    df.rename(columns={
        "Time": "frame.time",
        "Source": "ip.src",
        "Destination": "ip.dst",
        "Length": "frame.len"
    }, inplace=True)

    # Convert time column
    df["frame.time"] = pd.to_datetime(
        df["frame.time"],
        errors='coerce'
    )

    # Remove rows with invalid time
    df = df.dropna(subset=["frame.time"])

    # Create 5-second time bins
    df["time_bin"] = (
        df["frame.time"].astype("int64") // 10**9
    )

    # Simplified flow ID
    df["flow_id"] = (
        df["ip.src"].astype(str) + "-" +
        df["ip.dst"].astype(str) + "-" +
        df["time_bin"].astype(str)
    )

    # Aggregate flows
    flows = df.groupby("flow_id").agg({
        "frame.len": ["count", "sum", "mean"],
        "frame.time": ["min", "max"]
    })

    # Rename columns
    flows.columns = [
        "packet_count",
        "total_bytes",
        "avg_packet_size",
        "start_time",
        "end_time"
    ]

    flows = flows.reset_index()

    # Calculate duration
    flows["duration"] = (
        flows["end_time"] - flows["start_time"]
    ).dt.total_seconds()

    # Add label
    flows["Label"] = label

    # Save output
    output_file = file.replace(".csv", "_flows.csv")
    flows.to_csv(output_file, index=False)

    print(f"✔ Saved: {output_file}\n")


# File → Label mapping
files = {
    "scan1.csv": "PortScan",
    "dos1.csv": "DoS",
    "brute1.csv": "BruteForce",
    "brute2.csv": "BruteForce",
    "normal1.csv": "Normal",
    "normal2.csv": "Normal",
    "normal3.csv": "Normal"
}

# Process all files
for file, label in files.items():
    if os.path.exists(file):
        process_file(file, label)
    else:
        print(f"⚠ File not found: {file}")