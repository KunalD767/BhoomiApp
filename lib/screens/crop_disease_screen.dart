import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:permission_handler/permission_handler.dart';
import 'result_screen.dart';
import 'config.dart';

class CropDiseaseScreen extends StatefulWidget {
  const CropDiseaseScreen({super.key});

  @override
  _CropDiseaseScreenState createState() => _CropDiseaseScreenState();
}

class _CropDiseaseScreenState extends State<CropDiseaseScreen> {
  List<File> _images = [];
  bool _isLoading = false;
  List<Map<String, dynamic>> _previousScans = [];
  int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _loadPreviousScans();
  }

  Future<void> _requestPermissions() async {
    if (await Permission.storage.request().isGranted) {
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Storage permission is required to save and load scans.'),
        ),
      );
    }
  }

  Future<void> _pickImages() async {
    final picker = ImagePicker();
    final pickedFiles = await picker.pickMultiImage(); // Allow multiple image selection

    if (pickedFiles != null && pickedFiles.isNotEmpty) {
      setState(() {
        _images = pickedFiles.map((file) => File(file.path)).toList();
        _isLoading = true;
      });

      _uploadAndAnalyzeImages(_images);
    }
  }

  Future<void> _uploadAndAnalyzeImages(List<File> images) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);

    // Add each image to the multipart request
    for (var i = 0; i < images.length; i++) {
      request.files.add(await http.MultipartFile.fromPath('images', images[i].path, filename: 'image_$i.jpg'));
    }

    try {
      final response = await request.send().timeout(const Duration(seconds: 120));
      final responseData = await response.stream.toBytes();
      final result = jsonDecode(String.fromCharCodes(responseData));

      if (result.containsKey('error')) {
        throw Exception(result['error']);
      }

      String disease = result['disease'];
      String insights = result['insights'];

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultScreen(
            disease: disease,
            insights: insights,
            images: _images, // Pass the list of images
          ),
        ),
      );

      // Save the scan details for each image
      for (var image in _images) {
        _savePreviousScan(disease, insights, image.path);
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('An error occurred: $e'),
        duration: const Duration(seconds: 5),
        action: SnackBarAction(
          label: 'Retry',
          onPressed: () {
            _uploadAndAnalyzeImages(images);
          },
        ),
      ));
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _savePreviousScan(String disease, String insights, String imagePath) async {
    final prefs = await SharedPreferences.getInstance();
    final scanData = {
      'disease': disease,
      'insights': insights,
      'image': imagePath,
      'date': DateTime.now().toIso8601String(),
      'status': disease.toLowerCase().contains('healthy') ? 'Healthy' : 'Unhealthy',
    };
    _previousScans.add(scanData);
    prefs.setString('previous_scans', jsonEncode(_previousScans));
  }

  Future<void> _loadPreviousScans() async {
    final prefs = await SharedPreferences.getInstance();
    final scans = prefs.getString('previous_scans');
    if (scans != null) {
      setState(() {
        _previousScans = List<Map<String, dynamic>>.from(jsonDecode(scans));
      });
    }
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });

    switch (index) {
      case 0:
        Navigator.pushReplacementNamed(context, '/home'); 
        break;
      case 1:
        Navigator.pushReplacementNamed(context, '/chatbot'); 
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/account'); 
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.teal[100],
        elevation: 0,
        leading: Container(
          margin: const EdgeInsets.all(8.0),
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.3),
                blurRadius: 4,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: IconButton(
            icon: const Icon(Icons.arrow_back, color: Colors.black),
            onPressed: () {
              Navigator.pushReplacementNamed(context, '/home');
            },
          ),
        ),
        title: const Center(
          child: Text(
            'BHOOMI',
            style: TextStyle(color: Colors.black, fontSize: 24, fontWeight: FontWeight.bold),
          ),
        ),
        actions: const [SizedBox(width: 50)],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            const SizedBox(height: 20),
            const Text(
              'Upload your crop photos to identify diseases and receive tailored solutions instantly.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            _isLoading
                ? const CircularProgressIndicator()
                : _images.isEmpty
                    ? Image.asset(
                        'assets/images/crop_analysis_image.png',
                        height: 200,
                      )
                    : Expanded(
                        child: ListView.builder(
                          itemCount: _images.length,
                          itemBuilder: (context, index) {
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 5),
                              child: Image.file(_images[index], height: 200),
                            );
                          },
                        ),
                      ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _pickImages,
              icon: const Icon(Icons.photo_library, color: Colors.white),
              label: const Text('Select Images', style: TextStyle(color: Colors.white)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.teal,
                padding: const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'Previous Reports',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: _previousScans.length,
                itemBuilder: (context, index) {
                  final scan = _previousScans[index];
                  final date = DateTime.parse(scan['date']);
                  final formattedDate = '${date.day} ${_getMonthName(date.month)}';
                  final statusColor = scan['status'] == 'Healthy' ? Colors.green : Colors.red;

                  return ListTile(
                    leading: ClipRRect(
                      borderRadius: BorderRadius.circular(8.0),
                      child: Image.file(File(scan['image']), width: 70, height: 70, fit: BoxFit.cover),
                    ),
                    title: Text(formattedDate, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(scan['disease'] ?? 'Unknown', style: const TextStyle(fontSize: 14)),
                        Text(
                          scan['status'] ?? '',
                          style: TextStyle(
                            fontSize: 14,
                            color: statusColor,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.black54),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => ResultScreen(
                            disease: scan['disease'],
                            insights: scan['insights'],
                            images: [File(scan['image'])],
                          ),
                        ),
                      );
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: Colors.teal[100],
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(30.0),
            topRight: Radius.circular(30.0),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.5),
              spreadRadius: 1,
              blurRadius: 10,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: _selectedIndex,
          onTap: _onItemTapped,
          backgroundColor: Colors.transparent,
          elevation: 0,
          iconSize: 36.0,
          items: [
            BottomNavigationBarItem(
              icon: Icon(Icons.home, color: _selectedIndex == 0 ? Colors.teal : Colors.black),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.chat, color: _selectedIndex == 1 ? Colors.teal : Colors.black),
              label: 'Chatbot',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.account_circle, color: _selectedIndex == 2 ? Colors.teal : Colors.black),
              label: 'Account',
            ),
          ],
          selectedItemColor: Colors.teal,
          unselectedItemColor: Colors.black,
          showSelectedLabels: true,
          type: BottomNavigationBarType.fixed,
        ),
      ),
    );
  }

  String _getMonthName(int month) {
    const monthNames = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return monthNames[month - 1];
  }
}
