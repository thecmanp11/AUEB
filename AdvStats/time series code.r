# Load Johnson and Jonhson data

require(stats); require(graphics)
mydatats1<- JohnsonJohnson
y <- mydatats1


# Create a time series object and plot
j=ts(y, frequency=4, start = c(1960,1))
j
plot(j, type="o", col="blue", lty="dashed")

# Compute logarithms of the J&J, and differences
lj=log(j)        # compute the logartithm of the J&J  
dlj=diff(lj)   # compute the first differences of the log of J&J

# Create a multiple plots 
par(mfrow=c(3,2))        # set up the graphics  
plot(j,type="l", col='red', lwd=1,main="Time Series plot of Johnson & Johnson", ylab="Quarterly earnings per share")
hist(j, nclass=15, main="Histogram of Johnson & Johnson")
plot(lj,type="l", col='red', lwd=1,main="Log of Johnson & Johnson", ylab="Log of Quarterly earnings per share")
hist(lj, nclass=15, main="Histogram of log of Johnson & Johnson")
plot(dlj,type="l", col='red', lwd=1,main="Differences of log of Johnson & Johnson")
hist(dlj, nclass=15, main="Histogram of differences of log of J&J")

# Focus on the differences of logarithms of J&J
shapiro.test(dlj)          # normality test
par(mfrow=c(2,1))         
hist(dlj, prob=TRUE, 15)    # histogram    
lines(density(dlj))             # smooth it - ?density for details 
qqnorm(dlj,main="Normal QQplot of dlj")      # normal Q-Q plot  
qqline(dlj)                                                              # add a line    

# Create Autocorrelation and partial autocorrelation plots
par(mfrow=c(3,2))        # set up the graphics  
acf(j, 48, main="ACF of J&J")        # autocorrelation function plot 
pacf(j, 48, main="PACF of J&J")    # partial autocorrelation function 
acf(lj, 48, main="ACF of log of J&J")        
pacf(lj, 48, main="PACF of log of J&J")      
acf(dlj, 48, main="ACF of differences of  log of J&J")       
pacf(dlj, 48, main="PACF of differences of  log of J&J")    

# Create Autocorrelation and partial autocorrelation plots
par(mfrow=c(3,2))        # set up the graphics  
acf(ts(j,freq=1), 48, main="ACF of J&J")        # autocorrelation function plot 
pacf(ts(j,freq=1), 48, main="PACF of J&J")    # partial autocorrelation function 
acf(ts(lj,freq=1), 48, main="ACF of log of J&J")        
pacf(ts(lj,freq=1), 48, main="PACF of log of J&J")      
acf(ts(dlj,freq=1), 48, main="ACF of differences of  log of J&J")       
pacf(ts(dlj,freq=1), 48, main="PACF of differences of  log of J&J")    

# Estimate MA(q) models
ma1fit=arima(dlj,order=c(0,0,1))
ma1fit

ma4fit=arima(dlj,order=c(0,0,4))
ma4fit    

ma4afit=arima(dlj,order=c(0,0,4),method=c("CSS"))
ma4afit    
ma4restricted=arima(dlj,order=c(0,0,4),fixed=c(0,0,0,NA,NA))
ma4restricted

# Estimate AR(p) models
ar1fit=arima(dlj,order=c(1,0,0))
ar1fit    

ar4fit=arima(dlj,order=c(4,0,0))
ar4fit    

# Estimate ARMA(p,q) models
arma41fit=arima(dlj,order=c(4,0,1))
arma41fit    

arma41restricted=arima(dlj,order=c(4,0,1),fixed=c(0,0,0,NA,NA,NA))
arma41restricted

# Diagnostic plots
arma41residuals=arma41restricted$residuals
arma41residuals
residuals=ts(arma41residuals, frequency=4, start = c(1960,2))
residuals
par(mfrow=c(3,2))        # set up the graphics  
acf(ts(residuals,freq=1), 48, main="ACF of residuals")        
pacf(ts(residuals,freq=1), 48, main="PACF of residuals") 
acf(ts(residuals^2,freq=1), 48, main="ACF of squared residuals")        
pacf(ts(residuals^2,freq=1), 48, main="PACF of squared residuals") 
qqnorm(residuals,main="Normal QQplot of residuals")  
qqline(residuals)  
# Forecasts
forecast=predict(arma41restricted,8)   
forecast     
UL=forecast$pred+forecast$se
LL=forecast$pred-forecast$se
# plot of forecasts with 1 s.e. 
minx = min(dlj,LL); maxx = max(dlj,UL) 
ts.plot(dlj, forecast$pred, xlim=c(1960,1982), ylim=c(minx,maxx)) 
lines(forecast$pred, col="red", type="o") 
lines(UL, col="blue", lty="dashed") 
lines(LL, col="blue", lty="dashed")

# Comments
ar1=arima(dlj,order=c(1,0,0))
ar1
dar1<-arima(lj, order=c(1,1,0))
dar1



library(forecast)
auto.arima(lj)
arima.sim(list(order = c(1,1,2), ma=c(0.32,0.47), ar=0.8), n = 50)