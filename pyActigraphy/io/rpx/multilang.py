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

# In American english
# N.B: I don't know if there are diff between the headers created by
# Respironics softwares installed in the UK or in Canada.
# If so, please submit a GIT issue with an example so that I can add a new
# dictionary
fields_us = {
    'Name': 'Identity',
    'Device': 'Actiwatch Type',
    'Device_id': 'Actiwatch Serial Number',
    'Data': 'Epoch-by-Epoch Data',
    'Start_date': 'Data Collection Start Date',
    'Start_time': 'Data Collection Start Time',
    'Period': 'Epoch Length'
}

fields = {'FR': fields_fr, 'US': fields_us}

# Dictionary of the required columns in the data 'section' of the input file

# In French
columns_fr = {
    'Date': 'Date',
    'Time': 'Heure',
    'Activity': 'Activité',
    'Marker': 'Marqueur',
    'White_light': 'Lumière blanche'
}

# In American english
# N.B: I don't know if there are diff between the column names created by
# Respironics softwares installed in the UK or in Canada.
# If so, please submit a GIT issue with an example so that I can add a new
# dictionary
columns_us = {
    'Date': 'Date',
    'Time': 'Time',
    'Activity': 'Activity',
    'Marker': 'Marker',
    'White_light': 'White Light'
}

columns = {'FR': columns_fr, 'US': columns_us}

# List of keys corresponding to countries where the dates are encoded with day
# first.
day_first = ['FR']
