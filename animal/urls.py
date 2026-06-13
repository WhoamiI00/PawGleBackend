from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView
from accounts.views import CookieTokenRefreshView

urlpatterns = [
    path("admin/", admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path("api/token/", TokenObtainPairView.as_view(), name="get_token"),
    # Cookie-aware refresh: reads the httpOnly refresh cookie instead of body.
    path("api/token/refresh/", CookieTokenRefreshView.as_view(), name="refresh"),
    path("api-auth/", include("rest_framework.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
