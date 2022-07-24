
from bs4 import BeautifulSoup
import pandas as pd
import os
from pathlib import Path
from selenium import webdriver
import os
import time
from datetime import datetime as dt
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
from webdriver_manager.chrome import ChromeDriverManager

acoes_list = ['PETR4']

option = webdriver.ChromeOptions()
option.add_argument("--headless")
driver = webdriver.Chrome(ChromeDriverManager().install(),  options=option)
