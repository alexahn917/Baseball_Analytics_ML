library(RSQLite)
library(dplyr)
library(msm)
library(xtable)
library(data.table)
library(tidyr)

target_pitcher <- "Clayton Kershaw"

db <- src_sqlite('../../DB/pitchRx_14_16.sqlite3')

#dbListFields(db$con, "atbat")

# Join the location and names table into a new que table.
pitch <- dbGetQuery(db$con, 'SELECT pitch_type, inning, count, on_1b, on_2b, on_3b, type_confidence,
                    num, gameday_link FROM pitch')

names <- dbGetQuery(db$con, 'SELECT date, pitcher AS pitcher_id, pitcher_name, batter AS batter_id, 
                    batter_name, num, b_height, gameday_link, home_team_runs, away_team_runs, o AS out, p_throws as pitch_rl, stand as bat_rl FROM atbat') #stand?

games <- dbGetQuery(db$con, 'SELECT gameday_link, home_team_id FROM game')
games$gameday_link <- paste('gid_',games$gameday_link, sep="")

que <- inner_join(pitch, filter(names, pitcher_name == target_pitcher),
                     by = c('num', 'gameday_link'))

que <- inner_join(que, games, by = c('gameday_link'))

pitchfx <- as.data.frame(collect(que))
pitchfx <- data.table(pitchfx[ do.call(order, pitchfx[ , c('gameday_link','inning', 'num') ] ), ])
pitchfx[, batter_num:=as.numeric(factor(num)), by=gameday_link]
pitchfx <- as.data.frame(pitchfx)

# Create pitcher_at_home field
pitcher_id <- as.numeric(pitchfx$pitcher_id[1])
sqlStatement <- paste('SELECT team_id as pitcher_team_id FROM player WHERE id = ', pitcher_id, 'LIMIT 1')
pitcher_team_id <- as.numeric(dbGetQuery(db$con, sqlStatement))
pitchfx$pitcher_at_home[pitchfx$home_team_id == pitcher_team_id] <- 1
pitchfx$pitcher_at_home[pitchfx$home_team_id != pitcher_team_id] <- -1

# Create a new field for the batting order number.
pitchfx$batter_num <- ifelse(pitchfx$batter_num %% 9 == 0, 9, (pitchfx$batter_num %% 9))
pitchfx$batter_num <- as.factor(pitchfx$batter_num)
pitchfx$pitch_type <- as.factor(pitchfx$pitch_type)

# Get # of outs
pitchfx$out <- pitchfx$out - 1
pitchfx$out[pitchfx$out == -1] <- 0

# compute score difference
pitchfx$score_diff <- (as.numeric(pitchfx$home_team_runs) - as.numeric(pitchfx$away_team_runs)) * pitchfx$pitcher_at_home
pitchfx <- pitchfx[!is.na(pitchfx$score_diff),]

# Cleaning up data df.
data <- as.data.frame(select(pitchfx, type_confidence, pitch_type, batter_num, pitch_rl, bat_rl, inning,
                             count, out, on_1b, on_2b, on_3b, score_diff, pitcher_at_home))

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

# convert pitch ball types into integer classes
data$pitch_type <- as.character(data$pitch_type)
data$pitch_type[data$pitch_type == 'FA'] <- 0
data$pitch_type[data$pitch_type == 'FF'] <- 1
data$pitch_type[data$pitch_type == 'FT'] <- 2
data$pitch_type[data$pitch_type == 'FC'] <- 3
data$pitch_type[data$pitch_type == 'FS'] <- 4
data$pitch_type[data$pitch_type == 'SI'] <- 5
data$pitch_type[data$pitch_type == 'SF'] <- 6
data$pitch_type[data$pitch_type == 'SL'] <- 7
data$pitch_type[data$pitch_type == 'CH'] <- 8
data$pitch_type[data$pitch_type == 'CB'] <- 9
data$pitch_type[data$pitch_type == 'CU'] <- 10
data$pitch_type[data$pitch_type == 'KC'] <- 11
data$pitch_type[data$pitch_type == 'KN'] <- 12
data$pitch_type[data$pitch_type == 'EP'] <- 13
data$pitch_type[data$pitch_type == 'IN'] <- 14

# retrieve previous pitch ball type
prev_pitch_type <- lag(data$pitch_type, 1)
data$prev_pitch_type <- prev_pitch_type
data[data$balls == 0 & data$strikes == 0, ]$prev_pitch_type <- -1
data <- data[!is.na(data$prev_pitch_type),]

# drop 'NA's under pitch_type
data <- data[!is.na(data$pitch_type),]

# select only the common/frequent pitch types ( > 10%)
pitch_labels <- levels(as.factor(data$pitch_type))
pitch_type_props <- rep(0,length(pitch_labels))
i=1
for (PT in pitch_labels) {
  pitch_type_props[i] = length(data$pitch_type[data$pitch_type==PT]) / length(data$pitch_type)
  i = i+1;
}
pitch_type_props <- melt(data.frame(pitch_labels, pitch_type_props))
pitch_type_props <- pitch_type_props[order(-pitch_type_props$value),]

# drop levels that are not useful (Under proportions of 0.10)
used_pitch_types <- pitch_type_props$pitch_labels[pitch_type_props$value > 0.10]
data <- data[data$pitch_type %in% used_pitch_types,]

# only use instances where type confidence is at least 0.90
data <- data[data$type_confidence > 0.90,]
data <- data[, !names(data) %in% "type_confidence"]

write.csv(data, file=paste("ETL pipeline/raw_data/",target_pitcher,".csv", sep=""), row.names = FALSE)

tail(data)

# ----------------FEATURES----------------------
# Pitch Type: [pitch_type(char)]
# Previous Pitch: [prev_pitch_type(char)]
# Player Stats: [batter_num(#), pitch_rl(char), batter_rl(char)]
# Game: [Innings(#), balls(#), strikes(#), on_1b(#), on_2b(#), on_3b(#), score_differencial(#)]

# Data: [pitch_type, prev_pitch_type, batter_num, pitch_rl, bat_rl, inning, balls, strikes, on_1b, on_2b, on_3b, score_diff