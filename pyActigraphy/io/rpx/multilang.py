# Dictionary of the available fields in the RPX header

# In French
fields_fr = {
    'Name': 'Identité',
    'Device': 'Type d\'Actiwatch',
    'Device_id': 'Numéro de série de l\'Actiwatch',
    'Data': 'Données période par période',
    'Start_date': 'Date de début de la collecte des données',
    'Start_time': 'Heure de début de la collecte des données',
    'Period': 'Longueur de la période'
}

# In English
# N.B: I don't know if there are diff between the headers created by
# Respironics softwares installed in the UK, in the US or in Canada.
# If so, please submit a GIT issue with an example so that I can add a new
# dictionary
fields_eng = {
    'Name': 'Identity',
    'Device': 'Actiwatch Type',
    'Device_id': 'Actiwatch Serial Number',
    'Data': 'Epoch-by-Epoch Data',
    'Start_date': 'Data Collection Start Date',
    'Start_time': 'Data Collection Start Time',
    'Period': 'Epoch Length'
}

fields = {
    'ENG_UK': fields_eng,
    'ENG_US': fields_eng,
    'FR': fields_fr
}

# Dictionary of the required columns in the data 'section' of the input file

# In French
columns_fr = {
    'Date': 'Date',
    'Time': 'Heure',
    'Activity': 'Activité',
    'Marker': 'Marqueur',
    'White_light': 'Lumière blanche'
}

# In English
columns_eng = {
    'Date': 'Date',
    'Time': 'Time',
    'Activity': 'Activity',
    'Marker': 'Marker',
    'White_light': 'White Light'
}

columns = {
    'ENG_UK': columns_eng,
    'ENG_US': columns_eng,
    'FR': columns_fr
}

day_first = {'ENG_UK': True, 'ENG_US': False, 'FR': True}
