# Create a connection to the local instance of REDIS
# FIRST: Ubuntu= redis-server
library("redux")
r <- redux::hiredis(
  redux::redis_config(
    host = "127.0.0.1", 
    port = "6379"))

setwd('C:\\Users\\Clayton Preston\\Documents\\AUEB\\BigDataArch\\RedMongAss\\RECORDED_ACTIONS')
em = read.csv( 'emails_sent.csv' , sep=',', header=TRUE)
md = read.csv( 'modified_listings.csv' , sep=',', header=TRUE)

##### TEST ##########_-------------------------------------------------------------


# r$SETBIT("SeptemberSales","0","1")
# EXAMPLE test
i = 1
toString(i)
r$SET("test",toString(i),"1")
r$FLUSHALL()

# Total Rows in Modification dataframe
nrow(md)
# Count in df to verify
JanNMod = (md$MonthID == 1) + (md$ModifiedListing == 1)
table(JanNMod)



##### TASK 1 ##########_-------------------------------------------------------------


# TASK 1.1
# SETBIT for each row in which Month is Jan and ModList is YES
for (row in 1:nrow(md)) {
  if (md[row, "MonthID"] == 1 && md[row, "ModifiedListing"] == 1) {
    r$SETBIT("ModificationsJanuary",toString(row),"1")
  }

}

# How to calculate total ModificationsJanuary
r$BITCOUNT("ModificationsJanuary")


# TASK 1.2
# How to perform a bitwise NOT operation
r$BITOP("NOT","NOTModificationsJanuary","ModificationsJanuary")

# How to display the results of the bitwise NOT operation
r$BITCOUNT("NOTModificationsJanuary")

# TOTAL ROWS
nrow(md)
r$BITCOUNT("NOTModificationsJanuary") + r$BITCOUNT("ModificationsJanuary")
length(unique(md$UserID))


# TASK 1.3
aggem <- aggregate(data = em, EmailID ~ UserID + MonthID, 
                   function(EmailID) length(unique(EmailID)))
head(aggem)                                     
md = merge(x = md, y = aggem, by = c("UserID","MonthID"), all.x = TRUE)
md$EmailID[is.na(md$EmailID)] <- 0

# JAN EMAILS
for (row in 1:nrow(md)) {
  if (md[row, "MonthID"] == 1 && md[row, "EmailID"] >= 1) {
    r$SETBIT("EmailsJanuary",toString(row),"1")
  }
  
}
r$BITCOUNT("EmailsJanuary")

# FEB EMAILS
for (row in 1:nrow(md)) {
  if (md[row, "MonthID"] == 2 && md[row, "EmailID"] >= 1) {
    r$SETBIT("EmailsFebruary",toString(row),"1")
  }
  
}
r$BITCOUNT("EmailsFebruary")

# MARCH EMAILS
for (row in 1:nrow(md)) {
  if (md[row, "MonthID"] == 3 && md[row, "EmailID"] >= 1) {
    r$SETBIT("EmailsMarch",toString(row),"1")
  }
  
}
r$BITCOUNT("EmailsMarch")

# Eval Total
# How to perform a bitwise AND operation
r$BITOP("AND","AtLstOneALLThree",c("EmailsJanuary","EmailsFebruary","EmailsMarch"))

# How to display the results of the bitwise AND operation
r$BITCOUNT("AtLstOneALLThree")


# TASK 1.4
r$BITOP("AND","JanNMArch",c("EmailsJanuary","EmailsMarch"))
r$BITOP("NOT","NOTFeb","EmailsFebruary")
r$BITOP("AND","JanNMArchNOTFeb",c("JanNMArch","NOTFeb"))
r$BITCOUNT("JanNMArchNOTFeb")



NEWINFO

OLD






##### TASK 2 ##########_-------------------------------------------------------------
