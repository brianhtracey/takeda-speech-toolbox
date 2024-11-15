
import json
import os


# infile = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/FTD-Metadata-Files/FTD-metadata_with_cleaned_Diagnosis.csv'
# a = pandas.read_csv(infile)

input_dir = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/AWS_transcripts/ftd_free'
#input_dir = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/AWS_transcripts/hlthy_story'
#input_dir = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/Healthy_Ageing_Data/picture_transcripts'
#input_dir = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/FTD_Data/monologue_transcripts'
#input_dir = '/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/FTD_Data/monologue_transcripts'

output_dir = os.path.join(input_dir, 'for_blabla')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for jfile in os.listdir(input_dir):
    if jfile.endswith(".json"):
        with open(os.path.join(input_dir, jfile)) as f:
            data = json.load(f)
        # get the complete transcript out of json
        r = data['results']['transcripts'][0]
        txt = r['transcript']

        # now write the txt to a file
        outfile = jfile.replace('.json', '.txt')
        with open(os.path.join(output_dir, outfile), "w") as text_file:
            text_file.write(txt)

