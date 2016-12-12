#Load required packages
library('DBI')
library('dplyr')
library('ggplot2')
library('pitchRx')
library('RSQLite')

db <- src_sqlite('pitchRx_db.sqlite3')
con = dbConnect(RSQLite::SQLite(), 'pitchRx_Kershaw.sqlite3')

dbListTables(con)

atbat <- filter(tbl(db, 'atbat'), date >= '2013-01-01' & date <= '2016-12-09' )

target_pitcher <- "Clayton Kershaw"

#Create list for player
Kershaw <- filter(atbat, pitcher_name == target_pitcher)

#Join atbat and pitch tables
pitches <- tbl(db, 'pitch') #tbl_sqlite

Kershaw_pitch <- inner_join(pitches, Kershaw, by = c('num', 'gameday_link')) #tbl_sqlite

#Collect Data
Clayton_Kershaw16 <- collect(Kershaw_pitch)

#See Summary
summary(Clayton_Kershaw16)

#Data Frame Objective: 
#[pitcherID(#), BatterID(#), CatcherID(#), Innings(#), Score(#), Bases(#), count(#-#), previous_pitch(C)]