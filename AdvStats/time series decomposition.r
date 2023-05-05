
install.packages(c("fpp","forecast"))
library(fpp)
data(ausbeer)
timeserie_beer = tail(head(ausbeer, 17*4+2),17*4-4)
plot(as.ts(timeserie_beer))

library(forecast)
trend_beer = ma(timeserie_beer, order = 4, centre = T)
plot(as.ts(timeserie_beer))
lines(trend_beer)
plot(as.ts(trend_beer))

detrend_beer = timeserie_beer - trend_beer
plot(as.ts(detrend_beer))


	
m_beer = t(matrix(data = detrend_beer, nrow = 4))
seasonal_beer = colMeans(m_beer, na.rm = T)
plot(as.ts(rep(seasonal_beer,16)))


	
random_beer = timeserie_beer - trend_beer - seasonal_beer
plot(as.ts(random_beer))

	
recomposed_beer = trend_beer+seasonal_beer+random_beer
plot(as.ts(recomposed_beer))
	
ts_beer = ts(timeserie_beer, frequency = 4)
decompose_beer = decompose(ts_beer, "additive")
 
plot(as.ts(decompose_beer$seasonal))
plot(as.ts(decompose_beer$trend))
plot(as.ts(decompose_beer$random))
plot(decompose_beer)

ts_beer = ts(timeserie_beer, frequency = 4)
stl_beer = stl(ts_beer, "periodic")
seasonal_stl_beer   <- stl_beer$time.series[,1]
trend_stl_beer     <- stl_beer$time.series[,2]
random_stl_beer  <- stl_beer$time.series[,3]
 
plot(ts_beer)
plot(as.ts(seasonal_stl_beer))
plot(trend_stl_beer)
plot(random_stl_beer)
plot(stl_beer)


#########################################################
### more details
##########################################################
require(fpp2)
autoplot(elecequip) + xlab("Year") + ylab("New Orders Index") + 
ggtitle("Electrical Equipment Manufacturing Index (Euro area, 2005 = 100)")

low1 = lowess(elecequip,f=2/3)
plot(elecequip)
lines(low1,lty=2,lwd=3,col="red")
low2 = lowess(elecequip,f=1/3)
lines(low2,lty=3,lwd=3,col="green")
low3 = lowess(elecequip,f=.1)
lines(low3,lty=4,lwd=3,col="blue")

Tt = low3$y
# Subtract the Tt from the original series leaving behind St+Rt
EE.minusTt = elecequip - Tt
autoplot(EE.minusTt) + xlab("Year") + ggtitle("Detrended Electrical Equipment Index")

seasonfit = tslm(EE.minusTt~season)
St = fitted(seasonfit)
Rt = EE.minusTt - St
# Rt = residuals(seasonfit) is the same result

components = cbind(elecequip,Tt,St,Rt)
autoplot(components,facet=T) + xlab("Year") + ggtitle("Seasonal Decomposition of Electrical Equipment Index")


yt = St + Tt + Rt
plot(yt,elecequip,xlab="yt = St + Tt + Rt",ylab="Original Series (yt)")
abline(0,1,lwd=2,col="red")

checkresiduals(seasonfit)


ee.decomp = decompose(elecequip)
autoplot(ee.decomp) + xlab("Year") + ggtitle("Classical Additive Decomposition of Electrical Equipment Index")

seasonfit = tslm(EE.minusTt~season)
St = fitted(seasonfit)
# Subtract St from yt to obtain seasonally adjusted time series.
elec.seasadj = elecequip - St
autoplot(elecequip) + xlab("Year") + ylab("Seasonally Adjust Elec Manu Index") + ggtitle("Seasonally Adjusted Electrical Manufacturing Index (2005 = 100)") +
  autolayer(elecequip,series="Unadjusted Series") +
  autolayer(elec.seasadj,lwd=1.2,series="Seasonally Adjusted") +
  guides(colour=guide_legend(title="Series"))



attributes(ee.decomp)

St.decompose = ee.decomp$seasonal
# Subtract St from yt to obtain seasonally adjusted time series.
elec.seasadj2 = elecequip - St.decompose
autoplot(elecequip) + xlab("Year") + ylab("Seasonally Adjust Elec Manu Index") + ggtitle("Seasonally Adjusted Electrical Manufacturing Index (2005 = 100)") +
  autolayer(elecequip,series="Unadjusted Series") +
  autolayer(elec.seasadj2,lwd=1.2,series="Seasonally Adjusted") +
  guides(colour=guide_legend(title="Series"))

ee.test = tail(elecequip,24)
ee.train = head(elecequip,171)
ee.STL = stl(ee.train,t.window=13,s.window="periodic",robust=TRUE)
ee.SA = seasadj(ee.STL)
ee.fc1 = naive(ee.SA,h=24)
# Forecast with No Seasonality Added Back!
autoplot(ee.fc1) + ylab("New Orders Index") + ggtitle("Naive Forecasts of Seasonally Adjusted Data")

# Add Back in Seasonality
ee.fc2 = forecast(ee.STL,method="naive",h=24)
autoplot(ee.fc2) + ylab("New Orders Index") + ggtitle("Naive Forecast with Seasonality Added Back")

ee.fc3 = forecast(ee.STL,method="rwdrift",h=24) 
autoplot(ee.fc3) + ylab("New Orders Index") + ggtitle("Drift Forecast with Seasonality Added Back")

accuracy(ee.fc2,ee.test) # Naive Method
accuracy(ee.fc3,ee.test) # Drift Method

# The table below contains these forecasts


