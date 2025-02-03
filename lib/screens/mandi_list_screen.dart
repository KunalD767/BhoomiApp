import 'package:flutter/material.dart';
import 'package:bhoomi/helpers/location_helper.dart';
import 'package:bhoomi/helpers/api_helper.dart';
import 'package:geolocator/geolocator.dart';

class MandiListScreen extends StatefulWidget {
  @override
  _MandiListScreenState createState() => _MandiListScreenState();
}

class _MandiListScreenState extends State<MandiListScreen> {
  late Future<List<dynamic>> _nearbyMandis;

  @override
  void initState() {
    super.initState();
    _fetchNearbyMandis();
  }

  Future<void> _fetchNearbyMandis() async {
    try {
      Position position = await LocationHelper.getCurrentLocation();
      setState(() {
        _nearbyMandis = ApiHelper.fetchNearbyData(position.latitude, position.longitude, 'mandi');
      });
    } catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Nearby Mandis'),
        backgroundColor: Colors.teal[100],
      ),
      body: FutureBuilder<List<dynamic>>(
        future: _nearbyMandis,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No mandis found nearby.'));
          } else {
            return ListView.builder(
              itemCount: snapshot.data!.length,
              itemBuilder: (context, index) {
                final mandi = snapshot.data![index];
                return ListTile(
                  title: Text(mandi['name']),
                  subtitle: Text('Pricing: ${mandi['price']}'),
                  onTap: () {
                    Navigator.pushNamed(context, '/details', arguments: mandi);
                  },
                );
              },
            );
          }
        },
      ),
    );
  }
}
