import gmplot
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time


def get_coordinates(city_name, retries=3, backoff_factor=0.3):
    geolocator = Nominatim(user_agent="city_visualizer", timeout=10)
    for attempt in range(retries):
        try:
            location = geolocator.geocode(city_name)
            if location:
                return (location.latitude, location.longitude)
            else:
                print(f"Coordinates for {city_name} not found.")
                return None
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoding error for {city_name}: {e}. Retrying...")
            time.sleep(backoff_factor * (2**attempt))
    print(f"Failed to get coordinates for {city_name} after {retries} retries.")
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
