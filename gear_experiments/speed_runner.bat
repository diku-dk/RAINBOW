set SDF=200

python .\speed_experiment.py %SDF% --steps 500 --z1 20 --z2 20 >> speed_experiment_%SDF%_20_20.txt
python .\speed_experiment.py %SDF% --steps 500 --z1 20 --z2 40 >> speed_experiment_%SDF%_20_40.txt
python .\speed_experiment.py %SDF% --steps 500 --z1 40 --z2 20 >> speed_experiment_%SDF%_40_20.txt
