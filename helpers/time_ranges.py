

import calendar
import datetime
import time
import numpy as np

from datetime import datetime, timedelta

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta
        

def get_range(date1, date2, interval):
	return [dt.strftime('%Y-%m-%d T%H:%M') for dt in 
	datetime_range(datetime(2016, 9, 1, 7), datetime(2016, 9, 1, 9+12), 
	timedelta(minutes=27.7))]

# Find nearest value in numpy array
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# Convert to unix timestamp
def getUnixTimestamp(humanTime,dateFormat):
    unixTimestamp = int(calendar.timegm(datetime.strptime(humanTime, dateFormat).timetuple()))
    return unixTimestamp

# Convert to human readable timestamp
def getHumanTimestamp(unixTimestamp, dateFormat):
    humanTimestamp = datetime.utcfromtimestamp(int(unixTimestamp)).strftime(dateFormat)
    return humanTimestamp