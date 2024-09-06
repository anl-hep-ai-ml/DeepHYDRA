from html.parser import HTMLParser
import pandas as pd
from datetime import datetime
import re

class AtlasRunsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_h3 = False
        self.in_td = False
        self.in_p = False
        self.in_header = False
        self.header_buffer = []
        self.data_buffer = []
        self.runs_data = []
        self.current_run = {'Run Number': None, 'start': None, 'end': None, 'duration': None}
        self.current_year = 1900


    def extract_year_from_href(self, href):
        pattern = r'/(\d{4})/'

        match = re.search(pattern, href)
        if match:
            year = int(match.group(1))
            return year
        else:
            return 1900

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and ('align', 'center') in attrs and ('class', 'lumitable') in attrs:
            self.in_table = True
        elif tag == 'h3':
            self.in_h3 = True
        elif tag == 'a' and self.in_h3:
            for attr in attrs:
                if attr[0] == 'href' and attr[1].startswith('./run.py?run='):
                    #print(attr[1])
                    self.current_run['Run Number'] = int(attr[1].split('=')[1])
        elif tag == 'td' and self.in_table:
            self.in_td = True 
        elif tag == 'th' and self.in_table:
            self.in_header = True
        elif tag == 'p':
            self.in_p = True
        
        if tag == 'a' and self.in_p and self.current_year != 1900:
            for attr in attrs:
                if attr[0] == 'href':
                    href = attr[1]
                    #print(href)
                    if 'DATAPREPARATION' in href:
                        year = self.extract_year_from_href(href)
                        if year:
                            self.current_year = year
                            break

    def handle_data(self, data):
        if self.in_table and self.in_td:
            self.data_buffer.append(data.strip())
            #print(self.data_buffer)
        elif self.in_table and self.in_header:
            self.header_buffer.append(data.strip())
            #print(self.header_buffer)

    def process_data(self):
        if len(self.header_buffer) >= 1:
            header = self.header_buffer[0]
            value = self.data_buffer[0]
            #print(header)
            if header == 'Start':
                start = self.parse_datetime(value)
                #print(start)
                #print(self.current_year)
                self.current_run['start'] = start
            elif header == 'End':
                end = self.parse_datetime(value)
                self.current_run['end'] = end
            elif header == 'Duration':
                duration = self.parse_duration(value)
                self.current_run['duration'] = duration
        self.data_buffer.clear()
        self.header_buffer.clear()

    def parse_datetime(self, date_str):
        year = self.current_year
        date_with_year_str = f"{date_str}, {year}"
        date_format = "%a %b %d, %H:%M %Z, %Y"
        datetime_obj = datetime.strptime(date_with_year_str, date_format)
        return datetime_obj

    def parse_duration(self, duration_str):
        #print(duration_str)
        if 'hrs' in duration_str and 'min' in duration_str:
            hours, minutes = int(duration_str.split('hrs, ')[0].split()[0]), int(duration_str.split('min')[0].split()[0])
            total_minutes = hours * 60 + minutes
        elif 'hr' in duration_str and 'min' in duration_str:
            hours, minutes = int(duration_str.split('hr, ')[0].split()[0]), int(duration_str.split('min')[0].split()[0])
            total_minutes = hours * 60 + minutes
        elif 'day' in duration_str and 'min' in duration_str:
            day, minutes = int(duration_str.split('day, ')[0].split()[0]), int(duration_str.split('min')[0].split()[0])
            total_minutes = day * 24 * 60 + minutes
        elif 'day' in duration_str and 'hr' in duration_str and 'min' in duration_str:
            day, hours, minutes = int(duration_str.split('day, ')[0].split()[0]), int(duration_str.split('hr, ')[0].split()[0]), int(duration_str.split('min')[0].split()[0])
            total_minutes = day * 24 * 60 + hours * 60 + minutes
        elif 'day' in duration_str and 'hrs' in duration_str and 'min' in duration_str:
            day, hours, minutes = int(duration_str.split('day, ')[0].split()[0]), int(duration_str.split('hrs, ')[0].split()[0]), int(duration_str.split('min')[0].split()[0])
            total_minutes = day * 24 * 60 + hours * 60 + minutes
        elif 'hrs' in duration_str:
            hours = int(duration_str.split('hrs')[0])
            total_minutes = hours * 60
        elif 'min' in duration_str:
            minutes = int(duration_str.split('min')[0])
            total_minutes = minutes
        else:
            total_minutes = 0
        return total_minutes

    def handle_endtag(self, tag):
        if tag == 'table' and self.in_table:
            self.in_table = False
            #print(self.current_run)
            if all(value is not None for value in self.current_run.values()):
                self.runs_data.append(self.current_run.copy())
            self.current_run = {'Run Number': None, 'start': None, 'end': None, 'duration': None}
        elif tag == 'td' and self.in_td:
            self.in_td = False
            self.process_data()
        elif tag == 'th' and self.in_header:
            self.in_header = False

    @property
    def runs(self):
        if not self.runs_data:
            return pd.DataFrame()

        first_start_run_replaced = False
        first_end_run_replaced = False
        for idx, run in enumerate(self.runs_data):
            if run['start'] and not first_start_run_replaced:
                start = run['start'].replace(year=self.current_year)
                self.runs_data[idx]['start'] = start
                first_start_run_replaced = True
            if run['end'] and not first_end_run_replaced:
                end = run['end'].replace(year=self.current_year)
                self.runs_data[idx]['end'] = end
                first_end_run_replaced = True
            if first_start_run_replaced and first_end_run_replaced:
                break

        df = pd.DataFrame(self.runs_data)
        df.set_index('Run Number', inplace=True)
        return df


            

#with open('../../atlas-data-summary-runs-2023.html', 'r') as file:
#    html_content = file.read()


# html_content = """
# <h3>Run <a href="./run.py?run=461002">461002</a></h3>
# <table align="center" class="lumitable">
# <tbody>
# <tr><th>Beam Energy</th><td>6800 GeV</td></tr>
# <tr><th>Bunches Colliding</th><td>3</td></tr>
# <tr><th>Beta*</th><td>654 m</td></tr>
# <tr><th>Start</th><td>Mon Sep 18, 17:12 CEST</td></tr>
# <tr><th>End</th><td>Tue Sep 19, 10:23 CEST</td></tr>
# <tr><th>Duration</th><td>17 hrs, 12 min</td></tr>
# <tr><th>First/Last Stable LB</th><td>593/1013</td></tr>
# <tr><th>First/Last Ready LB</th><td>593/1009</td></tr>
# <tr><th>Solenoid/Toroid</th><td>7.7/20.4</td></tr>
# </tbody></table>
# <h3>Run <a href="./run.py?run=461003">461003</a></h3>
# <table align="center" class="lumitable">
# <tbody>
# <tr><th>Beam Energy</th><td>6800 GeV</td></tr>
# <tr><th>Bunches Colliding</th><td>3</td></tr>
# <tr><th>Beta*</th><td>654 m</td></tr>
# <tr><th>Start</th><td>Mon Sep 18, 17:12 CEST</td></tr>
# <tr><th>End</th><td>Tue Sep 19, 10:23 CEST</td></tr>
# <tr><th>Duration</th><td>17 hrs, 12 min</td></tr>
# <tr><th>First/Last Stable LB</th><td>593/1013</td></tr>
# <tr><th>First/Last Ready LB</th><td>593/1009</td></tr>
# <tr><th>Solenoid/Toroid</th><td>7.7/20.4</td></tr>
# </tbody></table>
# """

#aruns_parser = AtlasRunsParser()
#aruns_parser.feed(html_content)
#aruns_parser.close()
#
#runs_df = aruns_parser.runs
#run_numbers_all = list(runs_df.index.values)
##runs_df = pd.DataFrame(aruns_parser.runs)
#print(runs_df)
#print(run_numbers_all)

