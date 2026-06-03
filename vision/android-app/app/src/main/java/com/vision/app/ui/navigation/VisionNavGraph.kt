package com.vision.app.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.vision.app.ui.screens.admin.AdminDashboardScreen
import com.vision.app.ui.screens.authentication.AuthenticationScreen
import com.vision.app.ui.screens.enrollment.FaceEnrollmentScreen
import com.vision.app.ui.screens.login.LoginScreen
import com.vision.app.ui.screens.logs.LogsScreen
import com.vision.app.ui.screens.models.ModelStatusScreen
import com.vision.app.ui.screens.registration.RegistrationScreen
import com.vision.app.ui.screens.settings.SettingsScreen
import com.vision.app.ui.screens.splash.SplashScreen
import com.vision.app.ui.screens.users.UserManagementScreen

object Routes {
    const val SPLASH = "splash"
    const val LOGIN = "login"
    const val ADMIN = "admin"
    const val REGISTER = "register"
    const val ENROLL = "enroll/{userId}"
    const val AUTH = "auth"
    const val USERS = "users"
    const val LOGS = "logs"
    const val SETTINGS = "settings"
    const val MODELS = "models"
    fun enroll(userId: Long) = "enroll/$userId"
}

@Composable
fun VisionNavGraph() {
    val nav = rememberNavController()
    NavHost(navController = nav, startDestination = Routes.SPLASH) {
        composable(Routes.SPLASH) { SplashScreen(onDone = { nav.navigate(Routes.LOGIN) { popUpTo(Routes.SPLASH) { inclusive = true } } }) }
        composable(Routes.LOGIN) { LoginScreen(onAdmin = { nav.navigate(Routes.ADMIN) }, onAuth = { nav.navigate(Routes.AUTH) }) }
        composable(Routes.ADMIN) {
            AdminDashboardScreen(
                onRegister = { nav.navigate(Routes.REGISTER) },
                onUsers = { nav.navigate(Routes.USERS) },
                onLogs = { nav.navigate(Routes.LOGS) },
                onModels = { nav.navigate(Routes.MODELS) },
                onSettings = { nav.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.REGISTER) {
            RegistrationScreen(
                onEnroll = { userId -> nav.navigate(Routes.enroll(userId)) },
                onCancel = { nav.popBackStack() },
            )
        }
        composable(
            Routes.ENROLL,
            arguments = listOf(navArgument("userId") { type = NavType.LongType }),
        ) { back ->
            val userId = back.arguments?.getLong("userId") ?: 0L
            FaceEnrollmentScreen(userId = userId, onDone = { nav.popBackStack(Routes.ADMIN, false) })
        }
        composable(Routes.AUTH) { AuthenticationScreen(onCancel = { nav.popBackStack() }) }
        composable(Routes.USERS) { UserManagementScreen(onBack = { nav.popBackStack() }) }
        composable(Routes.LOGS) { LogsScreen(onBack = { nav.popBackStack() }) }
        composable(Routes.SETTINGS) { SettingsScreen(onBack = { nav.popBackStack() }) }
        composable(Routes.MODELS) { ModelStatusScreen(onBack = { nav.popBackStack() }) }
    }
}
