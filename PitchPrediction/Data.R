#Load required packages
library('DBI')
library('dplyr')
library('ggplot2')
library('pitchRx')
library('sqldf')

db <- src_sqlite("pitchRx_615.sqlite3", create = TRUE)

#Set up a scrape that will write to the new db
scrape(start = "2016-06-15", end = "2016-06-15", connect = db$con)
#scrape(start = "2016-06-01", end = '2016-06-30', connect = db$con)

#Download additional data and join
files <- c("inning/inning_hit.xml", "miniscoreboard.xml", "players.xml")
scrape(start = "2016-06-15", end = "2016-06-15", suffix = files, connect = db$con)

dbListTables(db$con)
db$con

#Updating
#dates <- collect(select(tbl(db, "game"), original_date))
#max.date <- max(as.Date(dates[!is.na(dates)], "%Y/%m/%d"))
# Append new PITCHf/x data
#scrape(start = max.date + 1, end = Sys.Date(), connect = db$con)
# Append other data
#scrape(start = max.date + 1, end = Sys.Date(), suffix = files, connect = db$con)