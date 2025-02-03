import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screens/splash_screen.dart';
import 'screens/language_selection_screen.dart';
import 'screens/welcome_screen.dart';
import 'screens/registration_screen.dart';
import 'screens/details_screen.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/crop_disease_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/product_details_screen.dart';
import 'screens/soil_analysis_screen.dart';
import 'screens/chatbot_screen.dart';
import 'screens/account_screen.dart';
import 'screens/user_data.dart';
import 'screens/mandi_transport_screen.dart';
import 'screens/warehouse_list_screen.dart';
import 'screens/warehouse_details_screen.dart';
import 'screens/mandi_list_screen.dart';
import 'screens/transport_list_screen.dart';
import 'screens/stubble_burning_screen.dart'; // Import new screen

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final prefs = await SharedPreferences.getInstance();
  bool isLoggedIn = prefs.getBool('isLoggedIn') ?? false;

  final String name = prefs.getString('name') ?? '';
  final String email = prefs.getString('email') ?? '';
  final String phoneNumber = prefs.getString('phoneNumber') ?? '';
  final String state = prefs.getString('state') ?? '';

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (_) => UserData()
            ..updateUserData(
              name: name,
              email: email,
              phoneNumber: phoneNumber,
              state: state,
            ),
        ),
      ],
      child: Bhoomi(isLoggedIn: isLoggedIn),
    ),
  );
}

class Bhoomi extends StatelessWidget {
  final bool isLoggedIn;

  Bhoomi({required this.isLoggedIn});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bhoomi',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      initialRoute: isLoggedIn ? '/home' : '/',
      routes: {
        '/': (context) => SplashScreen(),
        '/language-selection': (context) => LanguageSelectionScreen(),
        '/welcome': (context) => WelcomeScreen(),
        '/registration': (context) => RegistrationScreen(),
        '/details': (context) => DetailsScreen(),
        '/login': (context) => LoginScreen(),
        '/home': (context) {
          final userData = Provider.of<UserData>(context);
          return HomeScreen(
            name: userData.name,
            email: userData.email,
            phoneNumber: userData.phoneNumber,
            state: userData.state,
          );
        },
        '/crop-disease': (context) => CropDiseaseScreen(),
        '/profile': (context) => ProfileScreen(),
        '/product-details': (context) => ProductDetailsScreen(),
        '/soil-analysis': (context) => SoilAnalysisScreen(),
        '/chatbot': (context) => ChatbotScreen(),
        '/mandis-transport': (context) => MandisTransportScreen(),
        '/warehouse-list': (context) => WarehouseListScreen(),
        '/warehouse-details': (context) => WarehouseDetailsScreen(
          warehouse: ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>,
        ),
        '/mandi-list': (context) => MandiListScreen(),
        '/transport-list': (context) => TransportListScreen(),
        '/stubble-burning': (context) => StubbleBurningScreen(),
        '/account': (context) {
          final userData = Provider.of<UserData>(context);
          return AccountScreen(
            name: userData.name,
            email: userData.email,
            phoneNumber: userData.phoneNumber,
            state: userData.state,
          );
        },
      },
    );
  }
}
