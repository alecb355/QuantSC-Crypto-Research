# # Emthod #1: using solscan api key directly
# import requests
# from dotenv import load_dotenv
# from pathlib import Path
# import os
# import json

# # Get env vars set up
# dotenv_path = Path('.env')
# load_dotenv(dotenv_path=dotenv_path)
# solscan_api_key = os.getenv('SOLSCAN_API_KEY')

# # Endpoint configuration 
# endpoint = "https://pro-api.solscan.io/v2.0/account/transactions?limit=20"
# headers = {"token": solscan_api_key}

# # Get wallets
# with open('trading-strategy-1/example_wallet_list.json', 'r') as f:
#     wallets = json.load(f)

# # get transactions from a wallet
# def get_transactions_from_wallet(url, address, headers): 
#     response = requests.get(url + "?address="+address, headers=headers)
    
#     return response.json()

# # go through each wallet
# for wallet in wallets: 

#     # wallet info
#     wallet_address = wallet["address"]
#     wallet_type = wallet["type"]
    
#     print("Current wallet address: " + wallet_address)
    
#     # get 
#     transactions = get_transactions_from_wallet(endpoint, wallet["address"], headers)["data"]

#     for tx in transactions: 
#         token = "SOL"
#         time = tx.get("time")
#         tx_hash = tx.get("tx_hash")
#         tx_amount = tx.get("fee", 0) /  1e9
#         address_from = tx["signer"][0] if tx.get("signer") else None
#         address_to = wallet_address if address_from != wallet_address else None

#         json_obj = {
#             "timestamp": time,
#             "tx_signature": tx_hash,
#             "from_address": address_from,
#             "from_type": address_to,
#             "to_address": address_to,
#             "token": "SOL",
#             "amount": tx_amount
#         }

#         with open("all_transactions.json", "w") as f:
#             json.dump(transactions, f, indent=2)

# METHOD Number 2: Directly web scraping, takes much longer, unreliable 

import time
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd


driver = webdriver.Chrome()
driver.get("https://solscan.io/account/EGJnqcxVbhJFJ6Xnchtaw8jmPSvoLXfN2gWsY9Etz5SZ")
time.sleep(5)  # Initial load

all_data = []
target_rows = 400

while True:
    # Get current page data
    html = driver.page_source
    tables = pd.read_html(StringIO(html))
    
    if tables:
        all_data.append(tables[0])
        current_total = sum(len(df) for df in all_data)
        print(f"Scraped page with {len(tables[0])} rows. Total rows so far: {current_total}")
        
        # Check if we've reached or exceeded target
        if current_total >= target_rows:
            print(f"Reached target of {target_rows} rows")
            break
    
    try:
        # Find button by the exact class combination
        next_button = driver.find_element(By.XPATH, 
            "//button[contains(@class, 'h-[32px]') and contains(@class, 'w-[32px]') and .//svg[@class='lucide lucide-chevron-right']]")
        
        # Check if button is disabled by checking the class
        button_class = next_button.get_attribute("class")
        outer_html = next_button.get_attribute("outerHTML")
        
        # Check if disabled attribute exists or pointer-events-none is active
        if next_button.get_attribute("disabled") is not None:
            print("Next button is disabled - reached last page")
            break
        
        # Additional check: if the class contains disabled styles
        if "disabled:pointer-events-none" in button_class and "pointer-events-none" in button_class:
            print("Next button has disabled styling - reached last page")
            break
        
        # Click the button
        print("Clicking next button...")
        next_button.click()
        time.sleep(3)  # Wait for new data to load
        
    except NoSuchElementException:
        print("Next button not found")
        break
    except Exception as e:
        print(f"Error: {e}")
        break

driver.quit()

# Combine all data and take first 400 rows
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.drop_duplicates()  # Remove any duplicates
    final_df = final_df.head(400)  # Get exactly first 400 rows
    print(f"\nFinal dataset: {len(final_df)} rows")
    print(final_df.head())
else:
    print("No data scraped")