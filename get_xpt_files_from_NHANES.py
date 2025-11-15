import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

def download_nhanes_xpt_files(url, outdir):
    """
    Downloads all links to .xpt files from the specified NHANES webpage 
    and saves them to the local disk.
    """
    
    # 1. Define the target URL and download directory
    BASE_URL = url
    DOWNLOAD_DIR = outdir # Use a raw string for the Windows path

    print(f"Targeting URL: {BASE_URL}")
    print(f"Saving files to: {DOWNLOAD_DIR}\n")

    # 2. Create the target directory if it doesn't exist
    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print(f"Directory created or already exists: {DOWNLOAD_DIR}")
    except OSError as e:
        print(f"Error creating directory {DOWNLOAD_DIR}: {e}")
        return

    # 3. Fetch the webpage content
    try:
        response = requests.get(BASE_URL, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return

    # 4. Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 5. Find all links that end with '.xpt'
    xpt_links = []
    # Search for all <a> tags that have an 'href' attribute
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Check if the link ends with '.xpt' (case-insensitive)
        if href.lower().endswith('.xpt'):
            # Construct the absolute URL using urljoin to handle relative paths
            absolute_url = urljoin(BASE_URL, href)
            xpt_links.append(absolute_url)

    if not xpt_links:
        print("No .xpt files found on the webpage.")
        return

    print(f"Found {len(xpt_links)} .xpt file links. Starting download...\n")

    # 6. Download each file
    for i, download_url in enumerate(xpt_links):
        # Extract the filename from the URL path
        path = urlparse(download_url).path
        filename = os.path.basename(path)
        
        local_path = os.path.join(DOWNLOAD_DIR, filename)

        try:
            # Stream the file download for efficiency
            file_response = requests.get(download_url, stream=True, timeout=30)
            file_response.raise_for_status()

            # Write the file content to the local disk
            with open(local_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[{i+1}/{len(xpt_links)}] Successfully downloaded: {filename}")
        
        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{len(xpt_links)}] Error downloading {filename} from {download_url}: {e}")
        except IOError as e:
            print(f"[{i+1}/{len(xpt_links)}] Error saving file {filename} to {local_path}: {e}")

    print("\nDownload process complete.")


url_list_2015 = ['https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2015',
            'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2015',
            'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2015',
            'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2015',
            'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2015'
            ]
url_list_2017_2020 = ['https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2017-2020',
                      'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2017-2020',
                      'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2017-2020',
                      'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2017-2020',
                      'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2020'
                      ]
if __name__ == "__main__":
    for url in url_list_2015:
        download_nhanes_xpt_files(url= url, outdir=r"D:\NHANES\2015-2016")
    for url in url_list_2017_2020:
        download_nhanes_xpt_files(url= url, outdir=r"D:\NHANES\2017-2020")