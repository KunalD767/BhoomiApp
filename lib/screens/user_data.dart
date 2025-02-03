import 'package:flutter/foundation.dart';

class UserData extends ChangeNotifier {
  String _name = '';
  String _email = '';
  String _phoneNumber = '';
  String _state = '';

  String get name => _name;
  String get email => _email;
  String get phoneNumber => _phoneNumber;
  String get state => _state;

  void updateUserData({
    required String name,
    required String email,
    required String phoneNumber,
    required String state,
  }) {
    _name = name;
    _email = email;
    _phoneNumber = phoneNumber;
    _state = state;
    notifyListeners();
  }
}
