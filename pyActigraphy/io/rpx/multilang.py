# Dictionary of the available fields in the RPX header

# In French
fields_fr = {
    'Name': 'Identité',
    'Device': 'Type d\'Actiwatch',
    'Device_id': 'Numéro de série de l\'Actiwatch',
    'Data': 'Données période par période',
    'Start_date': 'Date de début de la collecte des données',
    'Start_time': 'Heure de début de la collecte des données',
    'Period': 'Longueur de la période',
    'NAN': 'NAN'
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
    'Period': 'Epoch Length',
    'NAN': 'NaN'
}

# In German
fields_ger = {
    'Name': 'Identität',
    'Device': 'Actiwatch-Typ',
    'Device_id': 'Actiwatch-Seriennummer',
    'Data': 'Daten nach Epochen',
    'Start_date': 'Startdatum der Datenerfassung',
    'Start_time': 'Startzeit der Datenerfassung',
    'Period': 'Epochenlänge',
    'NAN': 'kZ'
}

fields = {
    'ENG_UK': fields_eng,
    'ENG_US': fields_eng,
    'FR': fields_fr,
    'GER': fields_ger
}

# Dictionary of the required columns in the data 'section' of the input file

# In French
columns_fr = {
    'Line': 'Ligne',
    'Date': 'Date',
    'Time': 'Heure',
    'Activity': 'Activité',
    'Marker': 'Marqueur',
    'White_light': 'Lumière blanche',
    'Red_light': 'Lumière rouge',
    'Green_light': 'Lumière verte',
    'Blue_light': 'Lumière bleu',
    'Sleep/Wake': 'Sommeil/Éveil',
    'Mobility': 'Mobilité',
    'Interval Status': 'Statut de l’intervalle',
    'S/W Status': 'Statut Sommeil/Éveil'
}

# In English
columns_eng = {
    'Line': 'Line',
    'Date': 'Date',
    'Time': 'Time',
    'Off_Wrist': 'Off-Wrist Status',
    'Activity': 'Activity',
    'Marker': 'Marker',
    'White_light': 'White Light',
    'Red_light': 'Red Light',
    'Green_light': 'Green Light',
    'Blue_light': 'Blue Light',
    'Sleep/Wake': 'Sleep/Wake',
    'Mobility': 'Mobility',
    'Interval Status': 'Interval Status',
    'S/W Status': 'S/W Status'
}

# In German
columns_ger = {
    'Line': 'Zeile',
    'Date': 'Datum',
    'Time': 'Zeit',
    'Off_Wrist': 'Status „Nicht am Handgelenk“',
    'Activity': 'Aktivität',
    'Marker': 'Markierung',
    'White_light': 'Weißes Licht',
    'Red_light': 'Rotes Licht',
    'Green_light': 'Grünes Licht',
    'Blue_light': 'Blaues Licht',
    'Sleep/Wake': 'Schlaf/Wach',
    # 'Mobility': '??',
    'Interval Status': 'Intervallstatus',
    # 'S/W Status': '??'
}

columns = {
    'ENG_UK': columns_eng,
    'ENG_US': columns_eng,
    'FR': columns_fr,
    'GER': columns_ger
}

day_first = {'ENG_UK': True, 'ENG_US': False, 'FR': True, 'GER': True}
