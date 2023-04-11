import sys 
map = {}
sortedMap = {}
def parseReducerOutput(): 
    for line in sys.stdin:
        line = line.strip() 
        hour, ipAndCount = line.split(" ")
        hourNumber, minute = hour.split(":")
        hourNumber = int(hourNumber[1:])
        ip, count = ipAndCount.split('\t')
        count = int(count)

        if(map.get(hourNumber) is None):
            map[hourNumber] = [[ip,count]]
        else:
            map[hourNumber].append([ip,count])
    for key in sorted(map.keys()):
        sortedMap[key] = map[key]
    topThreeIPsInAnHour()
def topThreeIPsInAnHour():
    for h in map:
        top_ips = sorted(sortedMap[h], key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 ips for Hour:{h} are {top_ips}")


def databaseSearch(timeRange):
    startTime, endTime = timeRange.split("-")
    if(len(startTime) == 1):
        startTime = int("0" + startTime)
    else:
        startTime = int(startTime)
    
    if(len(endTime) == 1):
        endTime = int("0" + endTime)
    else:
        endTime = int(endTime)
    topThreeIPsInRange(startTime,endTime)

def topThreeIPsInRange(startTime,endTime):
    print(f"The top IPs within the given output range {startTime}-{endTime} are:")
    for i in range(startTime, endTime):
        if(sortedMap.get(i) is not None):
            top_ips = sorted(sortedMap[i], key=lambda x: x[1], reverse=True)[:3]
            print(f"Top 3 ips for Hour:{i} are {top_ips}")

    


if __name__ == "__main__":
    parseReducerOutput()
    #databaseSearch("3-10")
