import time
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# ActionChains are a way to automate low level interactions such as mouse movements, mouse button actions, key press, and context menu interactions.
# This is useful for doing more complex actions like hover over and drag and drop.
#from selenium.webdriver.common.action_chains import ActionChains   # for auto scroll

from webdriver_manager.chrome import ChromeDriverManager

URL_TARGET = 'http://www.yahoo.com/'
def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(URL_TARGET)
    driver.implicitly_wait(10)  # wait page load 
    elem = driver.find_element(By.NAME, 'p')   # find the search box
    elem.send_keys('mobile01' + Keys.RETURN)

    
    time.sleep(10)  
    driver.quit()

if __name__ == '__main__':
    main()
