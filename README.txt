Baseball Analytics Machine Learning Project.

This is a repository consisting analytical approaches on MLB games data using Machine Learning tools.


Content
———————————
(1) Pitch Prediction
(2) ERA Clustering
(3) Number of Wins Prediction (not yet shown)



How to Use
———————————
(1) Pitch Prediction.

	a) use ‘data_scrape.R’ to collect pitcher data for specified duration.
	b) change the name of the target pitcher in ‘Pitcher_Prediction_Modeling.R’ to start scraping target pitcher’s pitching instances up to past 3 years (2014 ~ 2016).

	ML library package:
	c.1) run ‘parseCSV.py’ to transform data into formatted ‘clean_data’
	d.1) run ’10-folding.R’ to sample data for cross validation
	e.1) run ‘run.sh’ under ‘model_code’ directory.

	Scikit-Learn Library:
	c.1) run ‘run.py’ under ‘scikit-learn’ directory.


If there is any question, please let me know at alexahn917@gmail.com