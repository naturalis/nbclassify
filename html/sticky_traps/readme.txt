### Sticky traps project.

This website was created as part of a longer term project to monitor the food availability for meadow bird chicks in the Netherlands.
Volunteers in this project deploy sticky traps in the field, and after 2 days pick them up, photograph them, and upload them to this website.
The website then automatically runs a script to analyze the images and record the results in a format usable for the researchers.
The script analyzing the images is analyse.py and it uses the settings file 'sticky_traps.yml' in the parent directory.
This settings file is in the parent directory because the code is initialized within the parent directory and the code looks there to find it.
The website is based heavily on the website developed earlier for the OrchID application, this code can be found on the master branch of this repo.

Some files that where not needed for this project where deleted to conserve space.
Models.py, Forms.py, Views.py, Url.py, and Templates.py are originally from the OrchID application, but heavily modified, so that most original code has changed.
analyse.py  and sticky_traps.yml are created by me, Ricardo Michels.
All other files in the project where not altered from the original OrchID application by much, if at all.

one dependency was added when compared to the original project.
This website uses django geoposition to enable the use of the google maps api to select the location where the traps were deployed.
