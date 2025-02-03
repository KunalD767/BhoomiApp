import 'package:flutter/material.dart';
import 'package:bhoomi/helpers/location_helper.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:geolocator/geolocator.dart';
import 'config.dart';

class TransportListScreen extends StatefulWidget {
  @override
  _TransportListScreenState createState() => _TransportListScreenState();
}

class _TransportListScreenState extends State<TransportListScreen> {
  late Future<List<dynamic>> _nearbyTransport; // Initialize as a Future

  @override
  void initState() {
    super.initState();
    _nearbyTransport = _fetchNearbyTransport(); // Set the Future
  }

  Future<List<dynamic>> _fetchNearbyTransport() async {
    try {
      Position position = await LocationHelper.getCurrentLocation();
      final transportList = await fetchTransportData(
          position.latitude, position.longitude);
      return transportList;
    } catch (e) {
      print(e);
      return [];
    }
  }

  Future<List<dynamic>> fetchTransportData(double latitude, double longitude) async {
    try {
      final url = Uri.parse('$baseUrl/transport'); // Replace with your backend URL
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: json.encode({"latitude": latitude, "longitude": longitude}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['recommendations'] != null && data['recommendations'] is List) {
          return data['recommendations'];
        } else {
          print("Unexpected response format: recommendations is not a list");
          return [];
        }
      } else {
        throw Exception("Failed to load transport data");
      }
    } catch (e) {
      print("Error fetching transport data: $e");
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Nearby Transport Services'),
        backgroundColor: Colors.teal[100],
      ),
      body: FutureBuilder<List<dynamic>>(
        future: _nearbyTransport,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            // Show a loading indicator while data is being fetched
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            // Show an error message if there is an issue
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            // Show a message if no transport services are found
            return Center(child: Text('No transport services found nearby.'));
          } else {
            // Display the transport services in a list
            return ListView.builder(
              itemCount: snapshot.data!.length,
              itemBuilder: (context, index) {
                final transport = snapshot.data![index];
                final distance = transport['distance']?.toStringAsFixed(2) ?? 'N/A';
                final rating = transport['rating']?.toString() ?? 'No rating available';

                return Container(
                  margin: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black),
                    borderRadius: BorderRadius.circular(10),
                    color: Colors.teal[50],
                  ),
                  child: ListTile(
                    title: Text(transport['name']),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Rating: $rating'),
                        Text('Distance: $distance km'),
                      ],
                    ),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: List.generate(
                        transport['rating'] != null ? transport['rating'].round() : 0,
                        (index) => Icon(Icons.star, color: Colors.yellow),
                      ),
                    ),
                    onTap: () {
                      Navigator.pushNamed(context, '/details', arguments: transport);
                    },
                  ),
                );
              },
            );
          }
        },
      ),
    );
  }
}
