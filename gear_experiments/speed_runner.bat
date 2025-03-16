set SDF=800
set STEPS=1000

python .\speed_experiment.py %SDF% --steps %STEPS% --z1 20 --z2 20 >> speed\speed_experiment_%SDF%_20_20.txt
python .\speed_experiment.py %SDF% --steps %STEPS% --z1 20 --z2 40 >> speed\speed_experiment_%SDF%_20_40.txt
python .\speed_experiment.py %SDF% --steps %STEPS% --z1 40 --z2 20 >> speed\speed_experiment_%SDF%_40_20.txt
