import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiHelper {
  static const String googleBaseUrl = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json';

  static Future<List<dynamic>> fetchNearbyData(double lat, double lon, String type) async {
    final response = await http.get(
      Uri.parse('$googleBaseUrl?location=$lat,$lon&radius=5000&type=$type&key=YOUR_GOOGLE_API_KEY'),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body)['results'];
    } else {
      throw Exception('Failed to load nearby data');
    }
  }
}
