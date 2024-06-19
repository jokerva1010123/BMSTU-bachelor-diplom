from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import threading
# Initialize Chrome WebDriver
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')
# options=chrome_options


def thread_function():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    # Open a website
    driver.get("https://thietkequancafe.com.vn/du-an-thi-cong-cong-trinh/")
    # Wait for page to load
    # time.sleep(1)
    # Fill data to form

    dress_field = driver.find_element(By.ID, 'fieldname3_1')
    cdt_field = driver.find_element(By.ID, 'fieldname19_1')
    email_field = driver.find_element(By.ID, 'fieldname22_1')
    sdt_field = driver.find_element(By.ID, 'fieldname26_1')
    dt_field = driver.find_element(By.ID, 'fieldname16_1')

    dress_field.send_keys("Hà Nội")
    cdt_field.send_keys("haha")
    email_field.send_keys("dangphucnguyen03099@gmail.com")
    sdt_field.send_keys("123456789")
    dt_field.send_keys("1000")
    # Find the element you want to click
    element = driver.find_element(By.CLASS_NAME, "pbSubmit")
    # Perform auto-click action using ActionChains
    actions = ActionChains(driver)
    actions.click(element).perform()
    # Wait for a few seconds to see the effect
    # Close the browser
    driver.quit()


for i in range(0, 10):
    thread = threading.Thread(target=thread_function)
    thread.start()
