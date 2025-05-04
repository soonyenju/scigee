import ee
import folium

class Canvas(object):
	def __init__(self):
		from folium import plugins
		self.cmap = {
			"veg": ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
					'66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
					'012E01', '011D01', '011301'
					],
			"grey": ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']
		}
		# Add custom basemaps to folium
		self.basemaps = {
			'Google Maps': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Maps',
				overlay = True,
				control = True
			),
			'Google Satellite': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Satellite',
				overlay = True,
				control = True
			),
			'Google Terrain': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Terrain',
				overlay = True,
				control = True
			),
			'Google Satellite Hybrid': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Satellite',
				overlay = True,
				control = True
			),
			'Esri Satellite': folium.TileLayer(
				tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
				attr = 'Esri',
				name = 'Esri Satellite',
				overlay = True,
				control = True
			)
		} 

	def draw(self, images, vis_params, layer_names = [], location=[20, 0], zoom_start = 3, height = 500):
		# draw using folium
		# Add EE drawing method to folium.
		folium.Map.add_ee_layer = self.add_ee_layer

		# Create a folium map object.
		self.m = folium.Map(location = location, zoom_start = zoom_start, height = height)

		# Add custom basemaps
		self.basemaps['Google Maps'].add_to(self.m)
		self.basemaps['Google Satellite Hybrid'].add_to(self.m)
		self.basemaps['Google Terrain'].add_to(self.m)

		# show coors interactively
		self.mouse_position()

		# # click show coors:
		# self._click_popup_coors()

		# click to place a popup marker
		self.m.add_child(folium.ClickForMarker(popup="Hello :)"))

		for image, vis_param, layer_name in zip(images, vis_params, layer_names):
			# Add the elevation model to the map object.
			self.m.add_ee_layer(image, vis_param, layer_name)

		# Add a layer control panel to the map.
		self.m.add_child(folium.LayerControl())

		# Add fullscreen button
		folium.plugins.Fullscreen().add_to(self.m)

		# # Display the map.
		# display(m)

	def save(self, filename):
		# example: m.save("filename.png")
		self.m.save(filename)

	# Define a method for displaying Earth Engine image tiles on a folium map.
	def add_ee_layer(self, ee_object, vis_params, name):
		
		try:    
			# display ee.Image()
			if isinstance(ee_object, ee.image.Image):    
				map_id_dict = ee.Image(ee_object).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
				).add_to(self.m)
			# display ee.ImageCollection()
			elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
				ee_object_new = ee_object.mosaic()
				map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
				).add_to(self.m)
			# display ee.Geometry()
			elif isinstance(ee_object, ee.geometry.Geometry):    
				folium.GeoJson(
				data = ee_object.getInfo(),
				name = name,
				overlay = True,
				control = True
			).add_to(self.m)
			# display ee.FeatureCollection()
			elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
				ee_object_new = ee.Image().paint(ee_object, 0, 2)
				map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
			).add_to(self.m)
		
		except Exception as e:
			print(e)
			print("Could not display {}".format(name))

	def mouse_position(self):
		formatter = "function(num) {return L.Util.formatNum(num, 3) + ' deg';};"

		folium.plugins.MousePosition(
			position='topright',
			separator=' | ',
			empty_string='NaN',
			lng_first=True,
			num_digits=20,
			prefix='Coordinates:',
			lat_formatter=formatter,
			lng_formatter=formatter,
		).add_to(self.m)

	def _click_popup_coors(self):
		folium.LatLngPopup().add_to(self.m)

	# easy visualizaiton parameter generator
	def viz(self, vmin, vmax, palette):
		return {
			"min": vmin,
			"max": vmax,
			"palette": palette
		}