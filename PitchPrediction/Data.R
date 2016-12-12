#Load required packages
library('DBI')
library('dplyr')
library('ggplot2')
library('pitchRx')
library('sqldf')

db <- src_sqlite("pitchRx_db.sqlite3", create = TRUE)

#Set up a scrape that will write to the new db
scrape(start = "2016-06-01", end = Sys.Date() - 1, connect = db$con)
#scrape(start = "2016-06-01", end = '2016-06-30', connect = db$con)

#Download additional data and join
files <- c("inning/inning_hit.xml", "miniscoreboard.xml", "players.xml")
scrape(start = "2016-06-01", end = Sys.Date() - 1, suffix = files, connect = db$con)

#update_db(src_sqlite, end = Sys.Date() - 1)
