from django.contrib import admin

from sorl.thumbnail.admin import AdminImageMixin

from orchid.models import Photo, Identity

class IdentityInline(admin.TabularInline):
    model = Identity
    extra = 0

class PhotoAdmin(AdminImageMixin, admin.ModelAdmin):
    list_display = ('file_name','id')
    inlines = [IdentityInline]

admin.site.register(Photo, PhotoAdmin)
