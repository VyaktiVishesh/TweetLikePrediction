import json
import csv
import os
import numpy as np
import random

# Define the JSON file name
# json_file = input("Enter the path to the JSON file: ")

# Define the CSV file name
csv_file = 'tweet_data3.csv'
count = 0
# Check if the CSV file exists
csv_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='', encoding='utf-8') as file:
# Open the CSV file in append mode
    writer = csv.writer(file)

    for i in range(900000):  #running loop for 900000 as then only we will have 300000 english tweets as they are aproximately one-third
        hour = random.randint(7,17)
        minu = random.randint(0,59)

        json_file = "Data/" + str(hour) + "/" + str(minu) + ".json"
        # Open the JSON file and read each line
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lineNumber = random.randint(0, len(lines))

            print("Tweet number: ", i)
            line = lines[lineNumber];
            # Load JSON object from the line
            json_data = json.loads(line)
            
            try:
                if "referenced_tweets" in json_data['data']:
                    text = json_data['includes']['tweets'][1]['text']
                    like_count = json_data['includes']['tweets'][1]['public_metrics']['like_count']
                    followers_count = json_data['includes']['users'][1]['public_metrics']['followers_count']
                    time_of_tweet = json_data['includes']['tweets'][1]['created_at']
                    language = json_data['includes']['tweets'][1]['lang']
                else:
                    text = json_data['data']['text']
                    like_count = json_data['data']['public_metrics']['like_count']
                    followers_count = json_data['includes']['users'][0]['public_metrics']['followers_count']
                    time_of_tweet = json_data['data']['created_at']
                    language = json_data['data']['lang']
                
                # Write the extracted data to the CSV file
                if language == 'en':
                    if not csv_exists:
                        writer.writerow(['Tweet Text', 'Like Count', 'Followers Count', 'Time of Tweet'])  # Write header
                        csv_exists = True  # Update the flag after writing the header
                    writer.writerow([text, like_count, followers_count, time_of_tweet])
            
            except (KeyError, IndexError) as e:
                # Increment count for missing or unexpected keys
                count += 1
                continue

print(f"Data from {json_file} has been appended to {csv_file}")
print(f"Number of missing or unexpected keys encountered: {count}")
