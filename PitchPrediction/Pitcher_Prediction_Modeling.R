library(RSQLite)
library(dplyr)
library(msm)
library(xtable)
library(data.table)
library(tidyr)

db <- src_sqlite('pitchRx_1.sqlite3')

# Join the location and names table into a new que table.
pitch <- dbGetQuery(db$con, 'SELECT pitch_type, inning, count, on_1b, on_2b, on_3b, type_confidence,
                    num, gameday_link FROM pitch')

#dbListTables(db$con)
#dbListFields(db$con, "atbat")
#dbListFields(db$con, "pitch")
#dbListFields(db$con, "player")
#dbListFields(db$con, "game")

#bso <- dbGetQuery(db$con, 'SELECT b, s, o FROM atbat')

names <- dbGetQuery(db$con, 'SELECT pitcher AS pitcher_id, pitcher_name, batter AS batter_id, 
                    batter_name, score, num, b_height, gameday_link, home_team_runs, away_team_runs, o AS out FROM atbat') #stand?

#head(dbGetQuery(db$con, 'SELECT * from game'))
#head(dbGetQuery(db$con, 'Select * from atbat'))

games <- dbGetQuery(db$con, 'SELECT gameday_link, home_team_id FROM game')
games$gameday_link <- paste('gid_',games$gameday_link, sep="")

pitcher_stats <- dbGetQuery(db$con, 'SELECT id as pitcher_id, rl AS pitch_rl, team_id as pitcher_team_id FROM player')
#era AS pitcher_era 

batter_stats <- dbGetQuery(db$con, 'SELECT id as batter_id, bats as bat_rl FROM player')
#                          rbi AS batter_rbi, 
#                           avg as batter_avg, 
#                           )

que <- inner_join(pitch, filter(names, pitcher_name == 'Clayton Kershaw'),
                     by = c('num', 'gameday_link'))

que <- inner_join(que, games, by = c('gameday_link'))
que <- inner_join(que, pitcher_stats, by = c('pitcher_id'))
que <- inner_join(que, batter_stats, by = c('batter_id'))


pitchfx <- as.data.frame(collect(que))
pitchfx <- data.table(pitchfx[ do.call(order, pitchfx[ , c('gameday_link','inning', 'num') ] ), ])
pitchfx[, batter_num:=as.numeric(factor(num)), by=gameday_link]
pitchfx <- as.data.frame(pitchfx)

# Create pitcher_at_home field
pitchfx$pitcher_at_home[pitchfx$home_team_id == pitchfx$pitcher_team_id] <- 1
pitchfx$pitcher_at_home[pitchfx$home_team_id != pitchfx$pitcher_team_id] <- -1

# Create a new field for the batting order number.
pitchfx$batter_num <- ifelse(pitchfx$batter_num %% 9 == 0, 9, (pitchfx$batter_num %% 9))
pitchfx$batter_num <- as.factor(pitchfx$batter_num)
pitchfx$pitch_type <- as.factor(pitchfx$pitch_type)

# Get # of outs
pitchfx$o <- pitchfx$o - 1
pitchfx$o[pitchfx$o == -1] <- 0

# Select most frequent pitch types
pitch_labels <- levels(pitchfx$pitch_type)
pitch_type_props <- rep(0,length(pitch_labels))
i=1
for (PT in pitch_labels) {
    pitch_type_props[i] = length(pitchfx$pitch_type[pitchfx$pitch_type==PT]) / length(pitchfx$pitch_type)
    i = i+1;
}
pitch_type_props <- melt(data.frame(pitch_labels, pitch_type_props))
pitch_type_props <- pitch_type_props[order(-pitch_type_props$value),]
#print(pitch_type_props)

# Drop levels that are not useful (Under proportions of 0.05)
used_pitch_types <- pitch_type_props$pitch_labels[pitch_type_props$value > 0.05]
pitchfx <- pitchfx[pitchfx$pitch_type == used_pitch_types,]

# compute score difference
pitchfx$score_diff <- (as.numeric(pitchfx$home_team_runs) - as.numeric(pitchfx$away_team_runs)) * pitchfx$pitcher_at_home



# Cleaning up data df.

#result <- select(data, pitch_type, prev_pitch_type, batter_num, pitch_rl, bat_rl, inning, balls, strikes, on_1b, on_2b, on_3b, score)
data <- as.data.frame(select(pitchfx, type_confidence, pitch_type, batter_num, pitch_rl, bat_rl, inning,
                             count, out, on_1b, on_2b, on_3b, score_diff))
#data <- as.data.frame(pitchfx[c(1:7,14, 17, 21:23)])
 
# compute score difference

# Only use instances where type confidence is at least 0.90
data <- data[data$type_confidence > 0.90,]
data <- data[, !names(data) %in% "type_confidence"]
#pitcher$uniqueID <- paste(pitcher$num, pitcher$gameday_link, pitcher$inning, sep='')

# convert handedness boolean (0|1)
data$pitch_rl[data$pitch_rl == 'L'] <- 0
data$pitch_rl[data$pitch_rl == 'R'] <- 1
data$bat_rl[data$bat_rl == 'L'] <- 0
data$bat_rl[data$bat_rl == 'R'] <- 1

#splot balls / strike counts into two columns
data <- separate(data = data, col = count, into = c("balls", "strikes"), sep = "\\-")

# convert on_bases IDs to boolean (0|1)
data$on_1b <- replace(data$on_1b, is.na(data$on_1b), 0)
data$on_1b <- replace(data$on_1b, data$on_1b != 0, 1)
data$on_2b <- replace(data$on_2b, is.na(data$on_2b), 0)
data$on_2b <- replace(data$on_2b, data$on_2b != 0, 1)
data$on_3b <- replace(data$on_3b, is.na(data$on_3b), 0)
data$on_3b <- replace(data$on_3b, data$on_3b != 0, 1)


# retrieve previous pitch ball type
prev_pitch_type <- lag(data$pitch_type, 1)
data$prev_pitch_type <- prev_pitch_type
data[data$balls == 0 & data$strikes == 0, ]$prev_pitch_type <- NA

#result <- select(data, pitch_type, prev_pitch_type, batter_num, pitch_rl, bat_rl, inning, balls, strikes, on_1b, on_2b, on_3b, score)


write.csv(data, file="output.csv")

data
#result

# ----------------FEATURES----------------------
# Pitch Type: [pitch_type(char)]
# Previous Pitch: [prev_pitch_type(char)]
# Player Stats: [batter_num(#), pitch_rl(char), batter_rl(char)]
# Game: [Innings(#), balls(#), strikes(#), on_1b(#), on_2b(#), on_3b(#), score_differencial(#)]

# Data: [pitch_type, prev_pitch_type, batter_num, pitch_rl, bat_rl, inning, balls, strikes, on_1b, on_2b, on_3b, score]

# Additionals:
# (1) previous pitch(es)
# (6) score_differencial(#)
# *Exclude type_confidence under .90*
# Change on_b as binary
# Potentially change to on base *bins*
# Steal Threat

# Changes:
# (2) batter_num // to replace qualitative measurement
# (3) catcher? for later
# (4) *take out pitcher's batting instances*
# (5) *take out ERA*

# Picther & Catcher combination?

# Max Schezer

# TODO
# (1) Sample Datasets
# (2) Modify SVM for multiclassification
# (3) Find external libraries for neural network
# (4) Come up with list of pitchers
# (5) Catchers?
