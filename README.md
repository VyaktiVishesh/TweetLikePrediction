# TweetLikePrediction
Steps to Run the Code:
1.) Download all the files and folders from main branch.
2.) Go to the directory Datasets and run the "data_extractor.py" which extracts all the tweets present in json format in "Data" directory.
3.) The extractor python file will generate the "tweet_data3.csv" in the same directory.
4.) Place that file in any directory and accordingly change the directories in the respective code of the models.
5.) Then run the models.

Steps to Run Interface:
1.) Install all respective libraries including "streamlit".
2.) Save the code and run command "streamlit run app.py" in terminal.
3.) The interface will run on local host.
Note: The model was trained on data of date "31-1-23" so interface may give slightly unhealthy data if tried on tweet of another day because of different trending topics on that day
