def extract_file_data(path):
    print(path)
    fileHandler = open(path, "r")

    timing_slack = []
    Start_point = []
    End_point = []
    while 1:
        line = fileHandler.readline()
        if not line:
            break
        # print(line.strip())
        #使用中严格遵照正则表达式使用，错一个空格都不行
        ret1 = re.match("Endpoint:",line)
        if ret1:
            tmp = re.findall(".*Endpoint:   \\\(.*) /.*", line)
            # print(tmp)
            end = tmp
            End_point.append(end)

        ret2 = re.match("Beginpoint:",line)
        if ret2:
            tmp = re.findall(".*Beginpoint: \\\(.*) /.*", line)
            start = tmp
            # print(start)
            Start_point.append(start)

        ret3 = re.match("= Slack Time", line)
        if ret3:
            tmp = re.findall(".*= Slack Time                   (.*).*", line)
            slack = tmp
            # print(slack)
            timing_slack.append(slack)
    # print(len(timing_slack))
    # print(len(Start_point))
    # print(len(End_point))
    fileHandler.close()
    return timing_slack,Start_point,End_point
