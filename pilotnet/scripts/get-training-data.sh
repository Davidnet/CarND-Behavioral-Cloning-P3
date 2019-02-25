set -euxo pipefail
declare -r URL="https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
declare -r DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
declare -r DESTINATION="$DIR/../../data/dataset.zip"
declare -r DESTINATIONFOLDER="$DIR/../../data"

echo "Saving data to: $DESTINATION"
curl $URL --create-dirs -o $DESTINATION

unzip $DESTINATION -d $DESTINATIONFOLDER