import gmplot
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="city_visualizer")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Coordinates for {city_name} not found.")
            return None
    except GeocoderTimedOut:
        print(f"Geocoding timed out for {city_name}.")
        return None


def visualize(cities_list):
    latitudes = []
    longitudes = []

    for city in cities_list:
        coords = get_coordinates(city)
        if coords:
            latitudes.append(coords[0])
            longitudes.append(coords[1])

    if not latitudes or not longitudes:
        print("No valid coordinates found.")
        return

    # Create the map plotter
    gmap = gmplot.GoogleMapPlotter(
        latitudes[0], longitudes[0], 7
    )  # Center the map around the first city

    # Plot the path
    gmap.plot(latitudes, longitudes, "blue", edge_width=2.5)

    # Draw the map to an HTML file
    gmap.draw("cities_path.html")
