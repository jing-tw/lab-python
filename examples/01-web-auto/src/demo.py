import selenium
from selenium import webdriver
from selenium.webdriver.support.select import Select
#from selenium.webdriver.common.action_chains import ActionChains   # for auto scroll
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

URL_TARGET = 'www.google.com'
def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(URL_TARGET)
    driver.implicitly_wait(10)


if __name__ == '__main__':
    main()
