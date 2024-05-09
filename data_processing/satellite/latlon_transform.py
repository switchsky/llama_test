def decimal_to_dms(latitude, longitude):
    def convert_to_dms(coord):
        degrees = int(coord)
        minutes_decimal = (coord - degrees) * 60
        minutes = int(minutes_decimal)
        seconds = (minutes_decimal - minutes) * 60
        return degrees, minutes, seconds

    lat_deg, lat_min, lat_sec = convert_to_dms(latitude)
    lon_deg, lon_min, lon_sec = convert_to_dms(longitude)

    lat_direction = 'N' if lat_deg >= 0 else 'S'
    lon_direction = 'E' if lon_deg >= 0 else 'W'

    lat_dms = "{}^{}^{}.{:04.4f}{}".format(abs(lat_deg), lat_min, int(lat_sec), lat_sec % 1 * 10000, lat_direction)
    lon_dms = "{}^{}^{}.{:04.4f}{}".format(abs(lon_deg), lon_min, int(lon_sec), lon_sec % 1 * 10000, lon_direction)

    return lat_dms, lon_dms

# 示例经纬度
latitude = 30.725968
longitude = 103.993028

result = decimal_to_dms(latitude, longitude)
print("纬度：{}".format(result[0]))
print("经度：{}".format(result[1]))