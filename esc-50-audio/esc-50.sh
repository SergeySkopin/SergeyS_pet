FILE=master.zip

if [ -f "$FILE" ]; then
	echo "$FILE exists."
	unzip -qq master.zip
else
    echo "$FILE does not exist. Starting download"
    wget "https://github.com/karoldvl/ESC-50/archive/master.zip"
    unzip -qq master.zip
fi
