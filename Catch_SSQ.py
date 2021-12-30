import time
import datetime
import requests
import json
import os
from bs4 import BeautifulSoup
import operator
import chardet
import re
import lxml
from html.parser import HTMLParser


def get_ssq_data():
    bet_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome69.0.3497.100 Safari537.36'}
    url = 'http://kaijiang.500.com/shtml/ssq/18121.shtml'
    try:
        lottery_req = requests.get(url, headers=bet_header, timeout=10)
        lottery_req.encoding = 'GB2312'
        # print(lottery_req.status_code)
        # print(lottery_req.text)
        soup = BeautifulSoup(lottery_req.text, 'lxml')
        # print(soup)
        tablesoup = soup.find_all('table', attrs={'class': 'kj_tablelist02'})
        # print(tablesoup)
        open_rows = tablesoup[0].findAll('div', attrs={'class': 'ball_box01'})
        # print(open_rows)
        open_tds = open_rows[0].findAll('li')
        red_ball1 = open_tds[0].get_text().strip()
        red_ball2 = open_tds[1].get_text().strip()
        red_ball3 = open_tds[2].get_text().strip()
        red_ball4 = open_tds[3].get_text().strip()
        red_ball5 = open_tds[4].get_text().strip()
        red_ball6 = open_tds[5].get_text().strip()
        blue_ball = open_tds[6].get_text().strip()
        print('open ball:%s,%s' % (red_ball1, red_ball2))
    except:
        print('Read timed out')
        return -1

    #
    # print(lottery_req.encoding)
    # print(lottery_req.apparent_encoding)


if __name__ == '__main__':
    get_ssq_data()
