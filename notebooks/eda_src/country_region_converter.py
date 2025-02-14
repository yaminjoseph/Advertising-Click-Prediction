import pandas as pd

region_map = {
    "Africa": 0,
    "Asia": 1,
    "Europe": 2,
    "North America": 3,
    "South America": 4,
    "Oceania": 5,
    "Middle East": 6,
    "Others": 7
}

# Region Dictionary
region_dic = {
    "Africa": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde",
        "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", 
        "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea", 
        "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", 
        "Guinea-Bissau", "Cote d'Ivoire", "Kenya", "Lesotho", "Liberia", "Libya", 
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", 
        "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", 
        "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", 
        "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", 
        "Zambia", "Zimbabwe", "Western Sahara",'Libyan Arab Jamahiriya'
    ],
    "Asia": [
        "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", 
        "Brunei Darussalam", "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", 
        "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyz Republic", 
        "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", 
        "Nepal", "North Korea", "Oman", "Palestinian Territory", "Qatar", "Saudi Arabia", 
        "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", 
        "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", 
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen", "Hong Kong", "Macao","Korea","Taiwan","Lao People's Democratic Republic",
        'Philippines','Pakistan'
    ],
    "Europe": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", 
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", 
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia", 
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", 
        "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", 
        "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", 
        "Norway", "Poland", "Portugal", "Romania", "Russian Federation", "San Marino", 
        "Serbia", "Slovakia (Slovak Republic)", "Slovenia", "Spain", "Sweden", "Switzerland", 
        "Turkey", "Ukraine", "United Kingdom", "Holy See (Vatican City State)", "Gibraltar",'Greenland','Macedonia'
    ],
    "North America": [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", 
        "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", 
        "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", 
        "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", 
        "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States of America",
        "United States Minor Outlying Islands", "Puerto Rico", "Guam", "Northern Mariana Islands",
        "American Samoa", "Cayman Islands", "Bermuda", "Isle of Man", "Saint Pierre and Miquelon"
    ],
    "South America": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", 
        "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"
    ],
    "Oceania": [
        "Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", 
        "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", 
        "Solomon Islands", "Tokelau", "Tonga", "Tuvalu", "Vanuatu", "Christmas Island",
        "Norfolk Island", "Pitcairn Islands", "New Caledonia", "French Polynesia", 
        "French Southern Territories", "Wallis and Futuna", "Cook Islands", "Bouvet Island (Bouvetoya)",
        "Heard Island and McDonald Islands", "Saint Helena", "Saint Barthelemy", 
        "Saint Martin", "Svalbard & Jan Mayen Islands","British Indian Ocean Territory (Chagos Archipelago)"
    ],
    "Middle East": [
        "Bahrain", "Cyprus", "Egypt", "Iran", "Iraq", "Israel", "Jordan", 
        "Kuwait", "Lebanon", "Oman", "Palestinian Territory", "Qatar", "Saudi Arabia", 
        "Syria", "Turkey", "United Arab Emirates", "Yemen",'Syrian Arab Republic'
    ],
    "Others": [
        "British Virgin Islands", "Bouvet Island (Bouvetoya)", "Aruba", "Saint Helena", 
        "Svalbard & Jan Mayen Islands", "Christmas Island", "Turks and Caicos Islands", 
        "Norfolk Island", "Cook Islands", "Faroe Islands", "Montserrat", "Wallis and Futuna", 
        "Jersey", "Antarctica (the territory South of 60 deg S)", "Western Sahara", 
        "Guernsey", "Martinique", "Falkland Islands (Malvinas)", "United States Virgin Islands", 
        "Kyrgyz Republic", "Brunei Darussalam", "Saint Pierre and Miquelon", 
        "French Southern Territories", "Greenland", "Guadeloupe", "French Guiana"
    ]
}

# Function 1: Region Function
def get_region(country):
    for region, countries in region_dic.items():
        if country in countries:
            return region
    return "Others" 
    